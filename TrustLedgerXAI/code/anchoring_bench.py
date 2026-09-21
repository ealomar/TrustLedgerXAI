"""
TrustLedger-XAI - anchoring-layer experiments (revision 2).

  E1  Digest determinism under alternative serialisations, on the 682 real
      decision-level artifacts and on a synthetic set.
  E2  Protocol conformance tests. These are deterministic checks that the
      ledger model enforces the stated semantics; they are reported as
      pass/fail, not as detection rates. Two tests exercise behaviour that can
      genuinely fail: concurrent first-writes to the same key (MVCC), and a
      generating organisation that controls a second endorsing organisation.
  E3  Cost of canonicalisation and hashing: median and IQR over repeated runs,
      on real artifacts and at larger synthetic sizes; measured world-state
      key and value sizes.
  E4  Merkle batching with an RFC 6962-style tree (leaf/node domain separation,
      no leaf duplication), and a demonstration of the two weaknesses of the
      previous construction.

    python aaer_pipeline.py       # writes artifacts.ndjson
    python anchoring_bench.py
"""

import copy
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import struct
import time

import rfc8785

from anchoring_ledger import MVCCLedger

random.seed(20260913)
OUT = "results"
os.makedirs(OUT, exist_ok=True)
REAL = [json.loads(l) for l in open("artifacts.ndjson")] if os.path.exists("artifacts.ndjson") else []
DEPLOY_N = 68_230          # every firm-year scored in the 2003-2014 test window


def sha(b):
    return hashlib.sha256(b).hexdigest()


def synthetic(n_attr, seed):
    rng = random.Random(seed)
    return {"assessment_id": f"SYN-{seed:07d}", "model_version": "trustledger-fused-2.0.0",
            "attribution_space": "fused_log_odds", "composite_score": rng.random(),
            "branch_contributions": {"cross_sectional": rng.uniform(-2, 2),
                                     "temporal": rng.uniform(-2, 2), "anomaly": rng.uniform(-1, 1)},
            "base_value": rng.uniform(-3, -1),
            "attributions": {f"f{i:03d}": rng.uniform(-1.5, 1.5) * 10 ** rng.uniform(-7, 0)
                             for i in range(n_attr)}}


def walk(o, f):
    if isinstance(o, float):
        return f(o)
    if isinstance(o, dict):
        return {k: walk(v, f) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(v, f) for v in o]
    return o


def naive(o, sort=True, round12=False):
    if round12:
        o = walk(o, lambda v: round(v, 12))
    return json.dumps(o, sort_keys=sort, separators=(",", ":")).encode()


def one_ulp(o):
    done = [False]
    def f(v):
        if done[0]:
            return v
        done[0] = True
        return struct.unpack("<d", struct.pack("<q", struct.unpack("<q", struct.pack("<d", v))[0] + 1))[0]
    return walk(copy.deepcopy(o), f)


# ----------------------------------------------------------------- E1
def e1():
    strategies = {"JCS / RFC 8785": lambda o: rfc8785.dumps(o),
                  "Sorted-key JSON": lambda o: naive(o),
                  "Unsorted JSON": lambda o: naive(o, sort=False),
                  "Sorted keys, rounded to 12 dp": lambda o: naive(o, round12=True)}
    sets = {"real": REAL, "synthetic": [synthetic(127, s) for s in range(200)]}
    rows = []
    for sname, arts in sets.items():
        for name, ser in strategies.items():
            res = {"reordered": 0, "text_roundtrip": 0, "ieee754_roundtrip": 0, "one_ulp_control": 0}
            for a in arts:
                d0 = sha(ser(a))
                items = list(a.items()); random.Random(1).shuffle(items)
                variants = {
                    "reordered": dict(items),
                    "text_roundtrip": walk(a, lambda v: float(repr(v))),
                    "ieee754_roundtrip": walk(a, lambda v: struct.unpack("<d", struct.pack("<d", v))[0]),
                    "one_ulp_control": one_ulp(a)}
                for k, v in variants.items():
                    res[k] += sha(ser(v)) == d0
            rows.append({"set": sname, "n": len(arts), "serialisation": name,
                         **{k: round(100 * v / len(arts), 1) for k, v in res.items()}})
    write("E1_determinism", rows)

    fp = f"py{platform.python_version()}|{platform.machine()}|{platform.system()}"
    ref = sha("".join(sha(rfc8785.dumps(a)) for a in REAL).encode())
    path = f"{OUT}/E1_crossversion.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["environment", "digest_over_real_artifacts"])
        w.writerow([fp, ref])


# ----------------------------------------------------------------- E2
def e2():
    orgs = ["AuditedCo", "AuditFirm", "Regulator"]
    rows = []

    def row(test, expected, observed):
        rows.append({"test": test, "expected": expected, "observed": observed,
                     "pass": expected == observed})

    arts = REAL[:200]
    L = MVCCLedger(orgs, "AuditedCo")
    for a in arts:
        L.submit(f"expl~{a['assessment_id']}", sha(rfc8785.dumps(a)), a["record_digest"],
                 "AuditedCo", ["AuditFirm"])
    L.cut_block()

    t = one_ulp(arts[0])
    row("Integrity: artifact altered by one ULP after anchoring", "verification fails",
        "verification fails" if sha(rfc8785.dumps(t)) != L.query(f"expl~{arts[0]['assessment_id']}")["H"]
        else "verification passes")
    try:
        L.submit(f"expl~{arts[0]['assessment_id']}", sha(rfc8785.dumps(t)), arts[0]["record_digest"],
                 "AuditedCo", ["AuditFirm"]); obs = "accepted"
    except PermissionError:
        obs = "rejected at endorsement"
    row("Non-substitution: re-anchor under an existing key", "rejected at endorsement", obs)
    obs = "verification fails" if sha(rfc8785.dumps(arts[1])) != L.query(
        f"expl~{arts[0]['assessment_id']}")["H"] else "verification passes"
    row("Non-substitution: another assessment's artifact presented for this key",
        "verification fails", obs)
    try:
        L.submit("expl~NEW-1", "00" * 32, "00" * 32, "AuditedCo", ["AuditedCo"]); obs = "accepted"
    except PermissionError:
        obs = "rejected at endorsement"
    row("Trust distribution: generating organisation self-endorses", "rejected at endorsement", obs)

    # concurrent first-writes: both endorsed before either commits
    races, invalid = 200, 0
    for i in range(races):
        Lr = MVCCLedger(orgs, "AuditedCo")
        Lr.submit("expl~RACE", sha(b"honest%d" % i), "00" * 32, "AuditedCo", ["AuditFirm"])
        Lr.submit("expl~RACE", sha(b"tampered%d" % i), "00" * 32, "AuditedCo", ["Regulator"])
        Lr.cut_block()
        invalid += Lr.invalid
        assert Lr.query("expl~RACE")["H"] == sha(b"honest%d" % i)
    row(f"Concurrent first-writes to one key ({races} races), MVCC validation on",
        "first-ordered commits, second invalid", "first-ordered commits, second invalid"
        if invalid == races else f"{invalid} invalidated")
    Ln = MVCCLedger(orgs, "AuditedCo"); Ln.validate_reads = False
    Ln.submit("expl~RACE", sha(b"honest"), "00" * 32, "AuditedCo", ["AuditFirm"])
    Ln.submit("expl~RACE", sha(b"tampered"), "00" * 32, "AuditedCo", ["Regulator"])
    Ln.cut_block()
    row("Same race with read-set validation disabled (chaincode check only)",
        "non-guarantee: last write wins",
        "non-guarantee: last write wins" if Ln.query("expl~RACE")["H"] == sha(b"tampered")
        else "first write kept")

    Lc = MVCCLedger(orgs, "AuditedCo", controlled_by_generator=["Regulator"])
    try:
        Lc.submit("expl~NEW-2", "00" * 32, "00" * 32, "AuditedCo", ["Regulator"]); obs = "accepted"
    except PermissionError:
        obs = "rejected"
    Lnaive = MVCCLedger(orgs, "AuditedCo")      # policy unaware of the control relationship
    try:
        Lnaive.submit("expl~NEW-3", "00" * 32, "00" * 32, "AuditedCo", ["Regulator"]); obs2 = "accepted"
    except PermissionError:
        obs2 = "rejected"
    row("Generator controls a second endorsing organisation, policy unaware of it",
        "non-guarantee: accepted", f"non-guarantee: {obs2}")
    row("Same, policy excludes organisations under the generator's control", "rejected", obs)

    Lf = MVCCLedger(orgs, "AuditedCo")
    fab = copy.deepcopy(REAL[0]); fab["attributions"] = {k: -v for k, v in fab["attributions"].items()}
    Lf.submit(f"expl~{fab['assessment_id']}", sha(rfc8785.dumps(fab)), fab["record_digest"],
              "AuditedCo", ["AuditFirm"]); Lf.cut_block()
    row("Fabricated rationale anchored at decision time, unbound verification",
        "non-guarantee: verification passes",
        "non-guarantee: verification passes" if sha(rfc8785.dumps(fab)) ==
        Lf.query(f"expl~{fab['assessment_id']}")["H"] else "verification fails")
    write("E2_conformance", rows)


# ----------------------------------------------------------------- E3
def time_per(arts, reps=30):
    can, hsh = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); blobs = [rfc8785.dumps(a) for a in arts]
        t1 = time.perf_counter(); [hashlib.sha256(b).digest() for b in blobs]
        t2 = time.perf_counter()
        can.append(1e6 * (t1 - t0) / len(arts)); hsh.append(1e6 * (t2 - t1) / len(arts))
    q = lambda xs: (statistics.median(xs), statistics.quantiles(xs, n=4)[0], statistics.quantiles(xs, n=4)[2])
    return q(can), q(hsh), int(statistics.median(len(rfc8785.dumps(a)) for a in arts))


def e3():
    rows = []
    sets = [("real (682 artifacts)", len(REAL[0]["attributions"]), REAL)]
    for n in [28, 250, 500]:
        sets.append((f"synthetic", n, [synthetic(n, s) for s in range(300)]))
    L = MVCCLedger(["A", "B"], "A")
    key_bytes = len(f"expl~{REAL[0]['assessment_id']}".encode())
    val_bytes = len(L.value_bytes(sha(b"x"), sha(b"y")))
    for label, n, arts in sets:
        (cm, c1, c3), (hm, h1, h3), size = time_per(arts)
        rows.append({"artifacts": label, "attributions": n, "artifact_bytes_median": size,
                     "canonicalise_us_median": round(cm, 1), "canonicalise_us_iqr": f"{c1:.1f}-{c3:.1f}",
                     "sha256_us_median": round(hm, 2), "sha256_us_iqr": f"{h1:.2f}-{h3:.2f}",
                     "world_state_key_bytes": key_bytes, "world_state_value_bytes": val_bytes})
    write("E3_cost", rows)


# ----------------------------------------------------------------- E4
def leaf_h(d):
    return hashlib.sha256(b"\x00" + d).digest()


def node_h(l, r):
    return hashlib.sha256(b"\x01" + l + r).digest()


def mth(leaves):
    """RFC 6962 Merkle Tree Hash over leaf data (here: explanation digests)."""
    n = len(leaves)
    if n == 1:
        return leaf_h(leaves[0])
    k = 1 << ((n - 1).bit_length() - 1)
    return node_h(mth(leaves[:k]), mth(leaves[k:]))


def path_len(n):
    return (n - 1).bit_length()


def legacy_root(leaves):
    lvl = list(leaves)
    while len(lvl) > 1:
        if len(lvl) % 2:
            lvl.append(lvl[-1])
        lvl = [hashlib.sha256(lvl[i] + lvl[i + 1]).digest() for i in range(0, len(lvl), 2)]
    return lvl[0]


def e4():
    leaves = [bytes.fromhex(sha(rfc8785.dumps(a))) for a in REAL]
    while len(leaves) < 2048:
        leaves += leaves[: 2048 - len(leaves)]
    L = MVCCLedger(["A", "B"], "A")
    per_commit = len(f"expl~{REAL[0]['assessment_id']}".encode()) + len(L.value_bytes(sha(b"x"), sha(b"y")))
    rows = []
    for bs in [1, 16, 64, 256, 1024, 2048]:
        ts = []
        for _ in range(50):
            t0 = time.perf_counter(); mth(leaves[:bs]); ts.append(1e3 * (time.perf_counter() - t0))
        commits = -(-DEPLOY_N // bs)
        rows.append({"batch_size": bs, "ledger_commits": commits,
                     "world_state_KB": round(commits * per_commit / 1024, 1),
                     "root_ms_median": round(statistics.median(ts), 3),
                     "inclusion_proof_hashes": path_len(bs)})
    write("E4_merkle_batching", rows)

    a, b, c = leaves[:3]
    lv = leaves[:8]
    parents = [hashlib.sha256(lv[i] + lv[i + 1]).digest() for i in range(0, 8, 2)]
    demo = [
        {"property": "Root of [a,b,c] equals root of [a,b,c,c]",
         "legacy": legacy_root([a, b, c]) == legacy_root([a, b, c, c]),
         "rfc6962": mth([a, b, c]) == mth([a, b, c, c])},
        {"property": "Internal nodes can be presented as leaves with the same root",
         "legacy": legacy_root(lv) == legacy_root(parents),
         "rfc6962": mth(lv) == mth(parents)},
    ]
    write("E4_merkle_weaknesses", demo)


def write(name, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n=== {name} ===")
    for r in rows:
        print(r)


if __name__ == "__main__":
    print(f"env: python {platform.python_version()} on {platform.system()}/{platform.machine()}")
    assert REAL, "run aaer_pipeline.py first"
    e1(); e2(); e3(); e4()
