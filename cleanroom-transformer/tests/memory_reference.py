"""Independent reference for the decision-memory commitments (SPEC.md 6), used by the tests.

Derived from the StandardSMT / IndexedSMT sketch this feature started from, with its defects fixed:
- verification walks the key bits in the same order as insertion (it used the reverse order, so
  honest non-membership proofs failed);
- a range-exclusion leaf must lie strictly before the window unless it is the genesis leaf
  (key <= low let an entry at exactly `low` be "proven" absent);
- inclusion is proven with an inclusion path (it reused the non-membership prover, which always
  raised for an existing key);
- keys are 256-bit id hashes, so ids never collide on a short path.
Uses hashlib.shake_256, not the engine's Keccak code, so the two implementations check each other.
"""
import hashlib

HL = 64
ID_DEPTH, TIME_DEPTH = 256, 64
TIME_END = 2**64 - 1


def shake(data: bytes, n: int = HL) -> bytes:
    return hashlib.shake_256(data).digest(n)


def h_leaf(data: bytes) -> bytes:
    return shake(b"\x00" + data)


def h_node(left: bytes, right: bytes) -> bytes:
    return shake(b"\x01" + left + right)


EMPTY = [h_leaf(b"EMPTY_LEAF_NODE")]
for _ in range(ID_DEPTH):
    EMPTY.append(h_node(EMPTY[-1], EMPTY[-1]))


def id_key(id_: str) -> int:
    return int.from_bytes(shake(b"\x03" + id_.encode(), 32), "big")


def smt_root(leaves: dict, depth: int) -> bytes:
    """Root of a sparse tree whose non-empty leaves are {key: leaf_hash}."""
    level = dict(leaves)
    for h in range(depth):
        parents = {}
        for k in {k >> 1 for k in level}:
            parents[k] = h_node(level.get(2 * k, EMPTY[h]), level.get(2 * k + 1, EMPTY[h]))
        level = parents
    return level.get(0, EMPTY[depth])


def climb(key: int, depth: int, leaf: bytes, siblings: dict) -> bytes:
    cur = leaf
    for h in range(depth):
        sib = siblings.get(h, EMPTY[h])
        cur = h_node(sib, cur) if (key >> h) & 1 else h_node(cur, sib)
    return cur


def time_leaf(key: int, value_hex: str, nxt: int) -> bytes:
    return h_leaf(f"{key}:{value_hex}:{nxt}".encode())


def roots(lines: list) -> dict:
    """lines: the memory file's lines (bytes, no newline). Checks the chain like the engine does."""
    import json
    hashes, latest, times = [], {}, []
    for i, line in enumerate(lines):
        e = json.loads(line)
        assert e["seq"] == i
        assert e["prev"] == (hashes[-1].hex() if hashes else "00" * HL)
        assert not times or e["time_us"] > times[-1]
        hashes.append(h_leaf(line))
        times.append(e["time_us"])
        latest[id_key(e["id"])] = hashes[-1]
    tleaves = {0: time_leaf(0, b"GENESIS".hex(), times[0] if times else TIME_END)}
    for i, t in enumerate(times):
        tleaves[t] = time_leaf(t, hashes[i].hex(), times[i + 1] if i + 1 < len(times) else TIME_END)
    r = {"id_root": smt_root(latest, ID_DEPTH), "time_root": smt_root(tleaves, TIME_DEPTH),
         "head": hashes[-1] if hashes else bytes(HL), "count": len(lines)}
    r["root"] = shake(b"\x02" + r["id_root"] + r["time_root"] + r["head"] + r["count"].to_bytes(8, "little"))
    return r


def verify(proof: dict) -> bool:
    """Independent verifier for engine proofs."""
    import json
    r = {k: bytes.fromhex(proof[k]) for k in ("id_root", "time_root", "head")}
    root = shake(b"\x02" + r["id_root"] + r["time_root"] + r["head"] + proof["count"].to_bytes(8, "little"))
    if root.hex() != proof["root"]:
        return False
    sibs = {h: bytes.fromhex(x) for h, x in proof["siblings"]}
    if proof["type"] in ("id-membership", "id-absence"):
        if proof["type"] == "id-membership":
            if json.loads(proof["entry"])["id"] != proof["id"]:
                return False
            leaf = h_leaf(proof["entry"].encode())
        else:
            leaf = EMPTY[0]
        return climb(id_key(proof["id"]), ID_DEPTH, leaf, sibs) == r["id_root"]
    if proof["type"] == "time-exclusion":
        leaf, lo, hi = proof["leaf"], proof["from"], proof["to"]
        if not ((leaf["key"] < lo or leaf["key"] == 0) and lo <= hi < leaf["next"]):
            return False
        return climb(leaf["key"], TIME_DEPTH, time_leaf(leaf["key"], leaf["value"], leaf["next"]), sibs) == r["time_root"]
    return False
