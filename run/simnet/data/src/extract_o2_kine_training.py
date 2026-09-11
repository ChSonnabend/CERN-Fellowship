#!/usr/bin/env python3
"""Extract tabular pruning-training data from O2 *_Kine.root files.

Run this inside an O2 ROOT environment, for example:

  apptainer exec -B /lustre -B /scratch /path/to/singularity_o2_al9.sif bash -lc \
    'eval $(alienv -w /scratch/alice/csonnab/MyO2/sw printenv O2PDPSuite::latest-o2); \
     python3 /lustre/alice/users/csonnab/cern-fellowship/run/simnet/data/src/extract_o2_kine_training.py'

The default input is the local stackPrune=false test production and the default
output is ../output/o2_kine_training.csv relative to this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


DEFAULT_INPUT = Path(
    "/lustre/alice/users/csonnab/PhD/jobs/simulation/sim_data/training/"
    "o2sim_09092026_anchoredMC_24arp2_559843/SC/0_100/500EV_1/tf1"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "output" / "o2_kine_training.csv"
)

NBITS_TRACK_ID = 31
NBITS_EVENT_ID = 19
NBITS_SOURCE_ID = 8
MASK_TRACK_ID = (1 << NBITS_TRACK_ID) - 1
MASK_EVENT_ID = (1 << NBITS_EVENT_ID) - 1
MASK_SOURCE_ID = (1 << NBITS_SOURCE_ID) - 1


FIELDNAMES = [
    "source_id",
    "event_id",
    "track_id",
    "mc_label_raw",
    "input_file",
    "pdg",
    "abs_pdg",
    "charge_sign",
    "mass",
    "energy",
    "ekin",
    "px",
    "py",
    "pz",
    "p",
    "pt",
    "eta",
    "phi",
    "theta",
    "rapidity",
    "vx",
    "vy",
    "vz",
    "t_ns",
    "event_vx",
    "event_vy",
    "event_vz",
    "dx_from_event",
    "dy_from_event",
    "dz_from_event",
    "r_xy",
    "r_from_event_xy",
    "r3_from_event",
    "mother_id",
    "second_mother_id",
    "first_daughter_id",
    "last_daughter_id",
    "direct_daughter_count",
    "has_daughters",
    "process",
    "status_code",
    "weight",
    "hit_mask",
    "num_detectors_with_hits",
    "has_hits",
    "to_be_done",
    "inhibited",
    "is_transported",
    "is_primary",
    "kept_by_o2",
    "can_avoid_geant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a CSV with MCTrack features and a GEANT-avoidance label. "
            "The target can_avoid_geant is 1 for tracks that O2 would not keep "
            "when pruning, unless they directly produce daughters."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="A *_Kine.root file or a directory containing *_Kine.root files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pattern",
        default="*_Kine.root",
        help="Glob used when --input is a directory.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional event limit per input file, useful for quick tests.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Optional total track limit across all files.",
    )
    parser.add_argument(
        "--source-id-start",
        type=int,
        default=0,
        help="Source id assigned to the first input file in mc_label_raw.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print a progress line after this many rows; set to 0 to disable.",
    )
    return parser.parse_args()


def import_root():
    try:
        import ROOT  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Could not import ROOT. Run this script inside the O2 environment "
            "used by the simulation job."
        ) from exc
    ROOT.gROOT.SetBatch(True)
    return ROOT


def input_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob(pattern))
        if files:
            return files
    raise SystemExit(f"No input kine files found for {path} with pattern {pattern!r}")


def mc_label_raw(track_id: int, event_id: int, source_id: int) -> int:
    return (
        (track_id & MASK_TRACK_ID)
        | ((event_id & MASK_EVENT_ID) << NBITS_TRACK_ID)
        | ((source_id & MASK_SOURCE_ID) << (NBITS_TRACK_ID + NBITS_EVENT_ID))
    )


def charge_sign(ROOT, pdg: int) -> int:
    if pdg == 0:
        return 0
    # Nuclear PDG code: +/- 10LZZZAAAI. Charge sign follows the ion sign.
    if abs(pdg) >= 1_000_000_000:
        z = (abs(pdg) // 10_000) % 1_000
        return 0 if z == 0 else (1 if pdg > 0 else -1)
    particle = ROOT.TDatabasePDG.Instance().GetParticle(pdg)
    if not particle:
        return 0
    charge = particle.Charge() / 3.0
    if charge > 0:
        return 1
    if charge < 0:
        return -1
    return 0


def safe_rapidity(energy: float, pz: float) -> float:
    denominator = energy - pz
    numerator = energy + pz
    if denominator <= 0.0 or numerator <= 0.0:
        return math.nan
    return 0.5 * math.log(numerator / denominator)


def event_vertex(tree) -> tuple[float, float, float]:
    header = getattr(tree, "MCEventHeader", None)
    if header is None:
        return 0.0, 0.0, 0.0
    try:
        return float(header.GetX()), float(header.GetY()), float(header.GetZ())
    except Exception:
        return 0.0, 0.0, 0.0


def status_code(tree, track_id: int) -> int:
    leaf = tree.GetLeaf("MCTrack.mStatusCode")
    if not leaf:
        return 0
    return int(leaf.GetValue(track_id))


def direct_daughter_counts(tracks) -> list[int]:
    counts = [0] * int(tracks.size())
    for track in tracks:
        mother = int(track.getMotherTrackId())
        if 0 <= mother < len(counts):
            counts[mother] += 1
    return counts


def make_row(ROOT, tree, tracks, counts, source_id: int, event_id: int, track_id: int, file_name: str) -> dict:
    track = tracks[track_id]
    px = float(track.Px())
    py = float(track.Py())
    pz = float(track.Pz())
    vx = float(track.Vx())
    vy = float(track.Vy())
    vz = float(track.Vz())
    t_ns = float(track.T())
    pt = math.hypot(px, py)
    p = math.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * math.log((p + pz) / (p - pz)) if p > abs(pz) else math.nan
    phi = math.atan2(py, px)
    theta = math.acos(pz / p) if p > 0 else math.nan
    mass = float(track.GetMass())
    energy = math.sqrt(max(0.0, mass * mass + p * p))
    evx, evy, evz = event_vertex(tree)
    dx = vx - evx
    dy = vy - evy
    dz = vz - evz
    direct_daughters = counts[track_id]
    first_daughter = int(track.getFirstDaughterTrackId())
    last_daughter = int(track.getLastDaughterTrackId())
    has_daughters = direct_daughters > 0 or first_daughter >= 0 or last_daughter >= 0
    kept_by_o2 = bool(track.getStore())
    can_avoid = (not kept_by_o2) and (not has_daughters)
    pdg = int(track.GetPdgCode())

    return {
        "source_id": source_id,
        "event_id": event_id,
        "track_id": track_id,
        "mc_label_raw": mc_label_raw(track_id, event_id, source_id),
        "input_file": file_name,
        "pdg": pdg,
        "abs_pdg": abs(pdg),
        "charge_sign": charge_sign(ROOT, pdg),
        "mass": mass,
        "energy": energy,
        "ekin": energy - mass,
        "px": px,
        "py": py,
        "pz": pz,
        "p": p,
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "theta": theta,
        "rapidity": safe_rapidity(energy, pz),
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "t_ns": t_ns,
        "event_vx": evx,
        "event_vy": evy,
        "event_vz": evz,
        "dx_from_event": dx,
        "dy_from_event": dy,
        "dz_from_event": dz,
        "r_xy": math.hypot(vx, vy),
        "r_from_event_xy": math.hypot(dx, dy),
        "r3_from_event": math.sqrt(dx * dx + dy * dy + dz * dz),
        "mother_id": int(track.getMotherTrackId()),
        "second_mother_id": int(track.getSecondMotherTrackId()),
        "first_daughter_id": first_daughter,
        "last_daughter_id": last_daughter,
        "direct_daughter_count": direct_daughters,
        "has_daughters": int(has_daughters),
        "process": int(track.getProcess()),
        "status_code": status_code(tree, track_id),
        "weight": float(track.getWeight()),
        "hit_mask": int(track.getHitMask()),
        "num_detectors_with_hits": int(track.getNumDet()),
        "has_hits": int(track.hasHits()),
        "to_be_done": int(track.getToBeDone()),
        "inhibited": int(track.getInhibited()),
        "is_transported": int(track.isTransported()),
        "is_primary": int(track.isPrimary()),
        "kept_by_o2": int(kept_by_o2),
        "can_avoid_geant": int(can_avoid),
    }


def main() -> int:
    args = parse_args()
    ROOT = import_root()
    files = input_files(args.input, args.pattern)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()

        for source_offset, path in enumerate(files):
            source_id = args.source_id_start + source_offset
            root_file = ROOT.TFile.Open(str(path))
            if not root_file or root_file.IsZombie():
                raise SystemExit(f"Could not open {path}")
            tree = root_file.Get("o2sim")
            if not tree:
                raise SystemExit(f"No o2sim tree in {path}")
            if not tree.GetBranch("MCTrack"):
                raise SystemExit(f"No MCTrack branch in {path}")

            n_events = int(tree.GetEntries())
            if args.max_events is not None:
                n_events = min(n_events, args.max_events)

            for event_id in range(n_events):
                tree.GetEntry(event_id)
                tracks = tree.MCTrack
                counts = direct_daughter_counts(tracks)
                for track_id in range(int(tracks.size())):
                    writer.writerow(
                        make_row(
                            ROOT,
                            tree,
                            tracks,
                            counts,
                            source_id,
                            event_id,
                            track_id,
                            path.name,
                        )
                    )
                    total_rows += 1
                    if args.progress_every and total_rows % args.progress_every == 0:
                        print(
                            f"wrote {total_rows} rows; current file={path.name} "
                            f"event={event_id}/{n_events}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if args.max_tracks is not None and total_rows >= args.max_tracks:
                        print(f"Wrote {total_rows} rows to {args.output}")
                        return 0

            root_file.Close()

    print(f"Wrote {total_rows} rows from {len(files)} file(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
