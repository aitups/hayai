#!/usr/bin/env python3
"""Minimal GGUF metadata/tensor-name dump (no gguf pip dep)."""
import struct
import sys
from pathlib import Path

GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16 = 0, 1, 2, 3
GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32 = 4, 5, 6
GGUF_TYPE_BOOL, GGUF_TYPE_STRING, GGUF_TYPE_ARRAY = 7, 8, 9
GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64 = 10, 11, 12


def read_str(buf, o):
    (n,) = struct.unpack_from("<Q", buf, o)
    o += 8
    s = buf[o : o + n].decode("utf-8", "replace")
    return s, o + n


def skip_val(buf, o, t):
    if t == GGUF_TYPE_UINT8 or t == GGUF_TYPE_INT8 or t == GGUF_TYPE_BOOL:
        return o + 1
    if t == GGUF_TYPE_UINT16 or t == GGUF_TYPE_INT16:
        return o + 2
    if t in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return o + 4
    if t in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return o + 8
    if t == GGUF_TYPE_STRING:
        _, o = read_str(buf, o)
        return o
    if t == GGUF_TYPE_ARRAY:
        (at, n) = struct.unpack_from("<IQ", buf, o)
        o += 12
        for _ in range(n):
            o = skip_val(buf, o, at)
        return o
    raise ValueError(f"unknown type {t}")


def read_val(buf, o, t):
    if t == GGUF_TYPE_UINT32:
        (v,) = struct.unpack_from("<I", buf, o)
        return v, o + 4
    if t == GGUF_TYPE_INT32:
        (v,) = struct.unpack_from("<i", buf, o)
        return v, o + 4
    if t == GGUF_TYPE_FLOAT32:
        (v,) = struct.unpack_from("<f", buf, o)
        return v, o + 4
    if t == GGUF_TYPE_UINT64:
        (v,) = struct.unpack_from("<Q", buf, o)
        return v, o + 8
    if t == GGUF_TYPE_BOOL:
        (v,) = struct.unpack_from("<B", buf, o)
        return bool(v), o + 1
    if t == GGUF_TYPE_STRING:
        return read_str(buf, o)
    if t == GGUF_TYPE_ARRAY:
        (at, n) = struct.unpack_from("<IQ", buf, o)
        o += 12
        vals = []
        for _ in range(n):
            v, o = read_val(buf, o, at)
            vals.append(v)
        return vals, o
    o2 = skip_val(buf, o, t)
    return f"<type{t}>", o2


def main(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"GGUF"
    ver = struct.unpack_from("<I", data, 4)[0]
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    o = 24
    print(f"ver={ver} tensors={n_tensors} kv={n_kv}")
    for _ in range(n_kv):
        key, o = read_str(data, o)
        (t,) = struct.unpack_from("<I", data, o)
        o += 4
        val, o = read_val(data, o, t)
        if any(
            x in key
            for x in (
                "architecture",
                "block",
                "cycle",
                "embed",
                "head",
                "context",
                "rope",
                "feed",
                "layer",
                "hrm",
                "scale",
            )
        ):
            print(f"META {key} = {val}")
    names = []
    for _ in range(n_tensors):
        name, o = read_str(data, o)
        (n_dims,) = struct.unpack_from("<I", data, o)
        o += 4
        dims = struct.unpack_from("<" + "Q" * n_dims, data, o)
        o += 8 * n_dims
        (ttype,) = struct.unpack_from("<I", data, o)
        o += 4
        (offset,) = struct.unpack_from("<Q", data, o)
        o += 8
        names.append(name)
    for n in names:
        if (
            "gate" in n
            or "init" in n
            or "z_l" in n
            or "embed" in n
            or n.startswith("blk.0.")
            or n.startswith("blk.15.")
            or n.startswith("blk.16.")
            or n.startswith("blk.31.")
            or n.startswith("output")
        ):
            print(f"T {n}")


if __name__ == "__main__":
    main(sys.argv[1])
