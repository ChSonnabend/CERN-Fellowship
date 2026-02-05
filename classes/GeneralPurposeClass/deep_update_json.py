import json
import collections

def deep_update(source, overrides, name="Dictionary", verbose = True):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value, verbose = False)
            source[key] = returned
        else:
            source[key] = overrides[key]
    if verbose:
        print("--- ", name, " ---")
        print(json.dumps(source, default=str, indent=4, sort_keys=False), "\n")
    return source