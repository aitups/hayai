#!/usr/bin/env python3
"""Convert a HuggingFace BART checkpoint to a Hayai-compatible GGUF (F16).

llama.cpp no longer ships a BART converter/runtime, so this one-shot helper maps
`BartForConditionalGeneration` weights to the `enc.blk.N.*` / `dec.blk.N.*` layout
consumed by `hayai-core/src/encoder_decoder_infer.rs` (`BartModel`).

Usage:  python scripts/bart_to_gguf.py <hf-model-id> <out.gguf> [F16|F32]
Requires: transformers, torch, gguf, huggingface_hub.
"""
import json
import os
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BartForConditionalGeneration
import gguf

MODEL = sys.argv[1] if len(sys.argv) > 1 else "facebook/bart-base"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/bart-base.F16.gguf"
DTYPE = (sys.argv[3] if len(sys.argv) > 3 else "F16").upper()
NP = np.float16 if DTYPE == "F16" else np.float32
GGML = gguf.GGMLQuantizationType.F16 if DTYPE == "F16" else gguf.GGMLQuantizationType.F32

print(f"loading {MODEL} -> {OUT} ({DTYPE})")
m = BartForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
sd = m.state_dict()
cfg = m.config
d = cfg.to_dict()

w = gguf.GGUFWriter(OUT, arch="bart")
w.add_name(os.path.basename(MODEL))
w.add_embedding_length(cfg.d_model)
w.add_feed_forward_length(cfg.encoder_ffn_dim)
w.add_block_count(cfg.encoder_layers)
w.add_decoder_block_count(cfg.decoder_layers)
w.add_head_count(cfg.encoder_attention_heads)
w.add_head_count_kv(cfg.encoder_attention_heads)
w.add_layer_norm_eps(d.get("layer_norm_eps", 1e-5))
w.add_context_length(cfg.max_position_embeddings)
w.add_bos_token_id(cfg.bos_token_id)
w.add_eos_token_id(cfg.eos_token_id)
w.add_pad_token_id(cfg.pad_token_id)
w.add_unk_token_id(3)
w.add_decoder_start_token_id(cfg.decoder_start_token_id)

# tokenizer (RoBERTa / GPT-2 byte-level BPE) from vocab.json + merges.txt
vp = hf_hub_download(MODEL, "vocab.json")
mp = hf_hub_download(MODEL, "merges.txt")
vocab = json.load(open(vp, encoding="utf-8"))
id_to_tok = {v: k for k, v in vocab.items()}
tokens = [""] * (max(id_to_tok) + 1)
for i, t in id_to_tok.items():
    tokens[i] = t
merges = []
with open(mp, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line and not line.startswith("#version"):
            merges.append(line)
special = {"<s>", "<pad>", "</s>", "<unk>", "<mask>"}
types = [2 if t == "<unk>" else (3 if t in special else 1) for t in tokens]
w.add_tokenizer_model("gpt2")
w.add_token_list(tokens)
w.add_token_merges(merges)
w.add_token_types(types)
w.add_add_bos_token(True)
w.add_add_eos_token(False)


def add(name, key):
    t = sd[key].to(torch.float32).contiguous().numpy().astype(NP)
    w.add_tensor(name, np.ascontiguousarray(t), raw_dtype=GGML)


add("token_embd.weight", "model.shared.weight")
for encdec in ("encoder", "decoder"):
    p = f"model.{encdec}"
    pre = "enc" if encdec == "encoder" else "dec"
    add(f"{pre}.pos_embd.weight", f"{p}.embed_positions.weight")
    add(f"{pre}.embed_norm.weight", f"{p}.layernorm_embedding.weight")
    add(f"{pre}.embed_norm.bias", f"{p}.layernorm_embedding.bias")
    nl = cfg.encoder_layers if encdec == "encoder" else cfg.decoder_layers
    for i in range(nl):
        b, g = f"{p}.layers.{i}", f"{pre}.blk.{i}"
        for proj in ("q", "k", "v", "out"):
            add(f"{g}.attn_{proj}.weight", f"{b}.self_attn.{proj}_proj.weight")
            add(f"{g}.attn_{proj}.bias", f"{b}.self_attn.{proj}_proj.bias")
        add(f"{g}.attn_norm.weight", f"{b}.self_attn_layer_norm.weight")
        add(f"{g}.attn_norm.bias", f"{b}.self_attn_layer_norm.bias")
        add(f"{g}.ffn_norm.weight", f"{b}.final_layer_norm.weight")
        add(f"{g}.ffn_norm.bias", f"{b}.final_layer_norm.bias")
        add(f"{g}.ffn_up.weight", f"{b}.fc1.weight")
        add(f"{g}.ffn_up.bias", f"{b}.fc1.bias")
        add(f"{g}.ffn_down.weight", f"{b}.fc2.weight")
        add(f"{g}.ffn_down.bias", f"{b}.fc2.bias")
        if encdec == "decoder":
            for proj in ("q", "k", "v", "out"):
                add(f"{g}.cross_attn_{proj}.weight", f"{b}.encoder_attn.{proj}_proj.weight")
                add(f"{g}.cross_attn_{proj}.bias", f"{b}.encoder_attn.{proj}_proj.bias")
            add(f"{g}.cross_attn_norm.weight", f"{b}.encoder_attn_layer_norm.weight")
            add(f"{g}.cross_attn_norm.bias", f"{b}.encoder_attn_layer_norm.bias")
flb = sd["final_logits_bias"].to(torch.float32).reshape(-1).numpy().astype(NP)
w.add_tensor("output.bias", np.ascontiguousarray(flb), raw_dtype=GGML)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("done:", OUT)
