#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import requests
from tqdm import tqdm

# Example usage:
# export GITHUB_TOKEN="...your token here..."
# python3.9 /scratch/csonnabe/cern-fellowship/LLM_finetune/fetch_prs.py --upstream https://github.com/AliceO2Group/O2Physics --authors ChSonnabend,njacazio,mhemmer,dsekihat,fgrosa,mfaggin --last-n 1000 --out o2_sft_last1000 --max-files-per-pr 20 --max-file-chars 20000

# -------------------------
# Helpers
# -------------------------
def strip_o2_license_header(text: str) -> str:
    start_of_file = True
    output = ""
    for line in text.splitlines():
        if "// Copyright" in line:
            start_of_file = True
        if start_of_file and line.startswith("/"):
            continue
        start_of_file = False
        output += line + "\n"
    return output.strip()
            

def parse_github_repo(url: str) -> Tuple[str, str]:
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not m:
        raise ValueError(f"Unsupported/invalid GitHub repo URL: {url}")
    return m.group(1), m.group(2)


def parse_prs(prs_arg: str) -> List[int]:
    parts = re.split(r"[,\s]+", prs_arg.strip())
    prs: List[int] = []
    for p in parts:
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"Invalid PR number: {p}")
        prs.append(int(p))
    if not prs:
        raise ValueError("No PR numbers provided.")
    return prs


def parse_authors(authors_arg: str) -> Set[str]:
    parts = [a.strip() for a in authors_arg.split(",") if a.strip()]
    if not parts:
        raise ValueError("No authors provided.")
    return {p.lower() for p in parts}


def gh_headers(token: Optional[str], accept: Optional[str] = None) -> Dict[str, str]:
    h = {
        "User-Agent": "pr-sft-dataset-builder/1.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    if accept:
        h["Accept"] = accept
    return h


def request_with_backoff(url: str, headers: Dict[str, str], timeout: int = 60) -> requests.Response:
    backoff = 2.0
    for _attempt in range(7):
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code in (200, 206):
            return r

        if r.status_code in (403, 429):
            retry_after = r.headers.get("Retry-After")
            reset = r.headers.get("X-RateLimit-Reset")
            if retry_after:
                sleep_s = float(retry_after)
            elif reset:
                now = time.time()
                sleep_s = max(1.0, float(reset) - now)
            else:
                sleep_s = backoff
            time.sleep(min(60.0, sleep_s))
            backoff *= 1.8
            continue

        if 500 <= r.status_code < 600:
            time.sleep(backoff)
            backoff *= 1.8
            continue

        return r

    return r


# -------------------------
# GitHub fetchers
# -------------------------
def fetch_pr(owner: str, repo: str, pr: int, token: Optional[str]) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}"
    r = request_with_backoff(url, headers=gh_headers(token), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"PR #{pr} metadata failed: {r.status_code} {r.text[:500]}")
    return r.json()


def fetch_pr_diff(owner: str, repo: str, pr: int, token: Optional[str]) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}"
    r = request_with_backoff(url, headers=gh_headers(token, accept="application/vnd.github.v3.diff"), timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"PR #{pr} diff failed: {r.status_code} {r.text[:500]}")
    return r.text


def fetch_pr_files(owner: str, repo: str, pr: int, token: Optional[str]) -> List[dict]:
    files: List[dict] = []
    page = 1
    per_page = 100
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}/files?per_page={per_page}&page={page}"
        r = request_with_backoff(url, headers=gh_headers(token), timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"PR #{pr} files failed: {r.status_code} {r.text[:500]}")
        chunk = r.json()
        if not chunk:
            break
        files.extend(chunk)
        if len(chunk) < per_page:
            break
        page += 1
    return files


def fetch_file_content_at_ref(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = request_with_backoff(url, headers=gh_headers(token, accept="application/vnd.github+json"), timeout=60)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Fetch content failed for {path}@{ref}: {r.status_code} {r.text[:500]}")

    data = r.json()
    if isinstance(data, dict) and data.get("type") == "file":
        encoding = data.get("encoding", "")
        content = data.get("content", "")
        if encoding == "base64" and content:
            raw = base64.b64decode(content)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="replace")

        dl = data.get("download_url")
        if dl:
            rr = request_with_backoff(dl, headers=gh_headers(token), timeout=120)
            if rr.status_code == 200:
                return rr.text
    return None


def list_recent_pr_numbers_by_authors(
    owner: str,
    repo: str,
    authors: Set[str],
    n: int,
    token: Optional[str],
) -> List[int]:
    """
    Walk PRs by created date (desc), filter by author login, return exactly N PR numbers (or raise if not enough).
    Uses the /pulls list endpoint (not search) to avoid the 1000-result cap of the search API.
    """
    collected: List[int] = []
    seen: Set[int] = set()

    page = 1
    per_page = 100

    # state=all so we include merged/closed PRs too
    while len(collected) < n:
        url = (
            f"https://api.github.com/repos/{owner}/{repo}/pulls"
            f"?state=all&sort=created&direction=desc&per_page={per_page}&page={page}"
        )
        r = request_with_backoff(url, headers=gh_headers(token, accept="application/vnd.github+json"), timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Listing PRs failed: {r.status_code} {r.text[:500]}")
        items = r.json()
        if not items:
            break  # no more PRs in repo

        for pr in items:
            num = pr.get("number")
            user = (pr.get("user") or {}).get("login", "")
            if not num or not user:
                continue
            if num in seen:
                continue
            seen.add(num)
            if user.lower() in authors:
                collected.append(int(num))
                if len(collected) >= n:
                    break

        page += 1

    if len(collected) < n:
        raise RuntimeError(
            f"Only found {len(collected)} PRs by authors {sorted(authors)} in {owner}/{repo}, "
            f"but you requested N={n}."
        )
    return collected


# -------------------------
# Dataset assembly
# -------------------------
@dataclass
class BuildOptions:
    max_files_per_pr: int
    max_file_chars: int
    include_deleted_files: bool
    include_renamed_as_old: bool
    exclude_regex: Optional[re.Pattern]
    include_extensions: Optional[set]
    keep_full_diff: bool


def should_include_file(path: str, status: str, opts: BuildOptions) -> bool:
    if opts.exclude_regex and opts.exclude_regex.search(path):
        return False
    if opts.include_extensions is not None:
        ext = Path(path).suffix.lower()
        if ext and ext not in opts.include_extensions:
            return False
    if status == "removed" and not opts.include_deleted_files:
        return False
    return True


def make_prompt(title: str, body: str) -> str:
    body = (body or "").strip()
    if body:
        return f"{title.strip()}\n\nPR description:\n{body}"
    return title.strip()


def make_context_block(files_context: List[Tuple[str, str]]) -> str:
    parts: List[str] = []
    for path, content in files_context:
        parts.append(f"### File: {path}\n```text\n{content}\n```\n")
    return "\n".join(parts).rstrip()


def main():
    ap = argparse.ArgumentParser(
        description="Build JSONL SFT dataset from GitHub PRs: input=PR meta + base file contents, output=PR diff."
    )
    ap.add_argument("--upstream", required=True, help="GitHub repo URL, e.g. https://github.com/OWNER/REPO")

    # Either provide PRs explicitly OR discover last N by authors
    ap.add_argument("--prs", default="", help="PR numbers, e.g. '123,124,130' or '123 124 130'")
    ap.add_argument("--authors", default="", help="Comma-separated GitHub logins to filter by (e.g. njacazio,mhemmer)")
    ap.add_argument("--last-n", type=int, default=0, help="If set (>0), auto-select exactly N most recent PRs by --authors")

    ap.add_argument("--out", default="out_dataset", help="Output directory")
    ap.add_argument("--jsonl", default="dataset.jsonl", help="JSONL filename within --out")

    ap.add_argument("--max-files-per-pr", type=int, default=30)
    ap.add_argument("--max-file-chars", type=int, default=20000)
    ap.add_argument(
        "--exclude",
        default=r"(^|/)(\.git|\.github|\.vscode)/|(\.png$|\.jpg$|\.jpeg$|\.gif$|\.pdf$|\.root$|\.zip$|\.tar$|\.gz$|\.so$|\.a$)",
        help="Regex of paths to exclude from context",
    )
    ap.add_argument(
        "--include-ext",
        default="",
        help="Comma-separated extensions to include (e.g. .cxx,.h,.hpp,.cmake,.txt). Empty means include all except excluded.",
    )
    ap.add_argument("--include-deleted", action="store_true")
    ap.add_argument("--include-renamed-old", action="store_true")
    ap.add_argument("--keep-full-diff", action="store_true")

    args = ap.parse_args()

    owner, repo = parse_github_repo(args.upstream)
    token = os.environ.get("GITHUB_TOKEN")  # recommended

    include_exts = None
    if args.include_ext.strip():
        include_exts = {e.strip().lower() for e in args.include_ext.split(",") if e.strip()}
        include_exts = {e if e.startswith(".") else f".{e}" for e in include_exts}

    opts = BuildOptions(
        max_files_per_pr=args.max_files_per_pr,
        max_file_chars=args.max_file_chars,
        include_deleted_files=args.include_deleted,
        include_renamed_as_old=args.include_renamed_old,
        exclude_regex=re.compile(args.exclude) if args.exclude else None,
        include_extensions=include_exts,
        keep_full_diff=args.keep_full_diff,
    )

    # Determine PR list
    if args.last_n and args.last_n > 0:
        if not args.authors.strip():
            raise ValueError("When using --last-n, you must also provide --authors.")
        authors = parse_authors(args.authors)
        pr_list = list_recent_pr_numbers_by_authors(owner, repo, authors, args.last_n, token)
        print(f"Selected {len(pr_list)} PRs (most recent by authors {sorted(authors)}).")
    else:
        if not args.prs.strip():
            raise ValueError("Provide either --prs or (--last-n and --authors).")
        pr_list = parse_prs(args.prs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save selected PR numbers for reproducibility
    (out_dir / "selected_prs.txt").write_text("\n".join(map(str, pr_list)) + "\n", encoding="utf-8")

    jsonl_path = out_dir / args.jsonl
    wrote = 0

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for pr in tqdm(pr_list):
            try:
                pr_data = fetch_pr(owner, repo, pr, token)
                diff_text = fetch_pr_diff(owner, repo, pr, token)
                files = fetch_pr_files(owner, repo, pr, token)

                title = pr_data.get("title", "")
                body = pr_data.get("body", "") or ""
                base_sha = (pr_data.get("base") or {}).get("sha", "")
                base_ref = (pr_data.get("base") or {}).get("ref", "")
                head_ref = (pr_data.get("head") or {}).get("ref", "")
                html_url = pr_data.get("html_url", "")
                author_login = ((pr_data.get("user") or {}).get("login", "") or "")

                contexts: List[Tuple[str, str]] = []
                included_files: List[dict] = []

                for f in files:
                    path = f.get("filename", "")
                    status = f.get("status", "")
                    if not path:
                        continue
                    if not should_include_file(path, status, opts):
                        continue
                    if len(contexts) >= opts.max_files_per_pr:
                        break

                    prev_path = f.get("previous_filename")
                    fetch_path = path
                    if status == "renamed" and opts.include_renamed_as_old and prev_path:
                        fetch_path = prev_path

                    content = fetch_file_content_at_ref(owner, repo, fetch_path, base_sha or base_ref, token)
                    if content is not None:
                        content = strip_o2_license_header(content)
                    else:
                        if status in ("added",):
                            content = "<FILE DOES NOT EXIST IN BASE (new file in PR)>"
                        else:
                            continue

                    if len(content) > opts.max_file_chars:
                        content = content[: opts.max_file_chars] + "\n<TRUNCATED>\n"

                    contexts.append((path, content))
                    included_files.append(
                        {
                            "filename": path,
                            "status": status,
                            "additions": f.get("additions", 0),
                            "deletions": f.get("deletions", 0),
                            "changes": f.get("changes", 0),
                        }
                    )

                record = {
                    "repo": f"{owner}/{repo}",
                    "pr_number": pr,
                    "pr_url": html_url,
                    "author": author_login,
                    "title": title,
                    "base_ref": base_ref,
                    "base_sha": base_sha,
                    "head_ref": head_ref,
                    "instruction": make_prompt(title, body),
                    "context": make_context_block(contexts),
                    "completion": diff_text,
                    "files_in_context": included_files,
                }

                pr_dir = out_dir / f"pr_{pr}"
                pr_dir.mkdir(parents=True, exist_ok=True)
                (pr_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "number": pr,
                            "title": title,
                            "url": html_url,
                            "author": author_login,
                            "base_ref": base_ref,
                            "base_sha": base_sha,
                            "head_ref": head_ref,
                            "files": included_files,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (pr_dir / "diff.patch").write_text(diff_text, encoding="utf-8")
                (pr_dir / "context.txt").write_text(record["context"], encoding="utf-8")

                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                wrote += 1
                print(f"OK PR #{pr}: {title}  (context files: {len(included_files)})")

            except Exception as e:
                print(f"ERROR PR #{pr}: {e}", file=sys.stderr)

    print(f"\nDone. Wrote {wrote}/{len(pr_list)} records to:")
    print(f"  {jsonl_path.resolve()}")
    if not token:
        print("\nTip: set GITHUB_TOKEN to avoid rate limits / access private repos.")


if __name__ == "__main__":
    main()