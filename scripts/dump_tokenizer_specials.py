#!/usr/bin/env python3
"""Inspect tokenizer special tokens + chat template of a GGUF."""
import struct
import sys

from pathlib import Path


def read_str(buf, o):
    (n,) = struct.unpack_from("<Q", buf, o)
    o += 8
    return buf[o : o + n].decode("utf-8", "replace"), o + n


def skip_val(buf, o, t):
    if t in (0, 1, 7):
        return o + 1
    if t in (2, 3):
        return o + 2
    if t in (4, 5, 6):
        return o + 4
    if t in (10, 11, 12):
        return o + 8
    if t == 8:
        _, o = read_str(buf, o)
        return o
    if t == 9:
        (at, n) = struct.unpack_from("<IQ", buf, o)
        o += 12
        for _ in range(n):
            o = skip_val(buf, o, at)
        return o
    raise ValueError(t)


def read_val(buf, o, t):
    if t in (4, 5):
        return struct.unpack_from("<i" if t == 5 else "<I", buf, o)[0], o + 4
    if t == 6:
        return struct.unpack_from("<f", buf, o)[0], o + 4
    if t == 10:
        return struct.unpack_from("<Q", buf, o)[0], o + 8
    if t == 7:
        return bool(buf[o]), o + 1
    if t == 8:
        return read_str(buf, o)
    if t == 9:
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
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    d = Path(path).read_bytes()
    n_t, n_kv = struct.unpack_from("<QQ", d, 8)
    o = 24
    for _ in range(n_kv):
        k, o = read_str(d, o)
        t = struct.unpack_from("<I", d, o)[0]
        o += 4
        v0 = o
        try:
            v, o = read_val(d, o, t)
        except Exception:
            o = v0
            o = skip_val(d, o, t)
            continue
        if k == "tokenizer.chat_template":
            print(f"TEMPLATE_START>>>{v}")
            print(f"<<<TEMPLATE_END")
        if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.special_tokens"):
            if isinstance(v, list):
                specials = [x for x in v if "<|" in str(x) or "<0x" in str(x) or "user" in str(x) or "assistant" in str(x)]
                print(f"{k}: {len(v)} tokens; specials={specials[:12]}")
            else:
                print(f"{k}: {str(v)[:200]}")


if __name__ == "__main__":
    main(sys.argv[1])
