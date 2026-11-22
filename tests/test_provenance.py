import json

from vglab.provenance import canonical_json_sha256, sha256_bytes, sha256_file


def test_sha256_bytes_is_hex_and_stable():
    digest = sha256_bytes(b"phase1")
    assert digest == sha256_bytes(b"phase1")
    assert digest != sha256_bytes(b"phase2")
    assert len(digest) == 64
    int(digest, 16)


def test_sha256_file_reads_bytes(tmp_path):
    path = tmp_path / "frame.bin"
    path.write_bytes(b"rgb-bytes")
    assert sha256_file(path) == sha256_bytes(b"rgb-bytes")


def test_canonical_json_sha256_ignores_key_order_and_rejects_nan():
    left = canonical_json_sha256({"b": 1, "a": [2, 3]})
    right = canonical_json_sha256({"a": [2, 3], "b": 1})
    assert left == right
    changed = canonical_json_sha256({"a": [2, 4], "b": 1})
    assert left != changed
    try:
        canonical_json_sha256({"depth": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("NaN must not enter a provenance hash")


def test_canonical_json_preserves_non_ascii():
    digest = canonical_json_sha256({"note": "米"})
    payload = json.dumps(
        {"note": "米"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert digest == sha256_bytes(payload)
