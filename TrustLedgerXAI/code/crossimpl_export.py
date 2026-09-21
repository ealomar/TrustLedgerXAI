"""
TrustLedger-XAI - E5 cross-implementation determinism, step 1 of 2 (Python).

    python crossimpl_export.py real        # the 682 real artifacts (artifacts.ndjson)
    python crossimpl_export.py stress      # synthetic stress set with edge-case floats
    node crossimpl_verify.mjs real|stress

The stress set deliberately places values at every point where Python and
JavaScript number formatting differ (exact zeros, negative zero, integral
floats, the 1e-6..1e-4 band, denormals, magnitudes >= 1e16). Agreement on it
is therefore a stress test of canonicalisation, not an estimate of how often
real artifacts diverge; the real set provides that estimate.
"""
import csv, hashlib, json, math, random, sys
import rfc8785

mode = sys.argv[1] if len(sys.argv) > 1 else "real"


def stress(seed):
    rng = random.Random(seed)
    return {"assessment_id": f"STRESS-{seed:05d}",
            "attributions": {f"f{i:03d}": rng.uniform(-1.5, 1.5) for i in range(28)},
            "edge": {"zero": 0.0, "neg_zero": -0.0, "integral": 2.0, "band": 2.5e-5,
                     "tiny": 5e-324, "large": 1e21, "boundary": 1e-6, "mantissa": 0.1 + 0.2}}


def floats(o):
    if isinstance(o, float):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from floats(v)
    elif isinstance(o, list):
        for v in o:
            yield from floats(v)


def divergent_features(a):
    """Float renderings on which Python json and JavaScript JSON.stringify differ."""
    kinds = set()
    for v in floats(a):
        av = abs(v)
        if v == 0.0 or (math.isfinite(v) and v == int(v) and av < 1e16):
            kinds.add("integral_or_zero")
        elif 1e-7 <= av < 1e-4 or 0 < av < 1e-7:
            kinds.add("small_magnitude")
        elif av >= 1e16:
            kinds.add("large_magnitude")
    return ";".join(sorted(kinds))


arts = ([json.loads(l) for l in open("artifacts.ndjson")] if mode == "real"
        else [stress(s) for s in range(200)])
with open(f"crossimpl_{mode}.ndjson", "w") as f:
    for a in arts:
        f.write(json.dumps(a) + "\n")
with open(f"digests_python_{mode}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "canonical", "naive", "divergent_features"])
    for i, a in enumerate(arts):
        w.writerow([i, hashlib.sha256(rfc8785.dumps(a)).hexdigest(),
                    hashlib.sha256(json.dumps(a, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    divergent_features(a)])
print(f"{mode}: wrote {len(arts)} artifacts and Python digests")
