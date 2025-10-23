"""
Precision-aware GPT copy that allows experimenting with MXFP4/NVFP4 dtypes.

The implementation mirrors nanochat.gpt.GPT but swaps the hard-coded
bf16 casts for configurable precision settings.
"""

from __future__ import annotations

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import (
    GPTConfig,
    Block,
    norm,
)
from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

from .dtypes import PrecisionConfig, ResolvedPrecision, resolve_precision_config


@dataclass
class PrecisionAwareGPTConfig:
    """Configuration for the experimental precision-aware GPT."""

    gpt: GPTConfig
    precision: PrecisionConfig = PrecisionConfig()


class PrecisionAwareGPT(nn.Module):
    """Copy of GPT that uses configurable precision settings."""

    def __init__(self, config: PrecisionAwareGPTConfig):
        super().__init__()
        self.config = config.gpt
        self.precision_cfg = resolve_precision_config(config.precision)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.vocab_size, self.config.n_embd),
                "h": nn.ModuleList(
                    [Block(self.config, layer_idx) for layer_idx in range(self.config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        # rotary embeddings cache (same strategy as the original implementation)
        self.rotary_seq_len = self.config.sequence_len * 10
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # -------------------------------------------------------------------------
    # Initialization helpers

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self._apply_precision_to_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _apply_precision_to_parameters(self):
        # First convert the entire module to compute precision.
        self.to(dtype=self.precision_cfg.compute)
        # Embedding / rotary buffers may use different dtypes.
        if self.precision_cfg.embedding != self.precision_cfg.compute:
            self.transformer.wte.weight.data = self.transformer.wte.weight.data.to(
                self.precision_cfg.embedding
            )
        if self.precision_cfg.rotary != self.precision_cfg.compute:
            self.cos = self.cos.to(self.precision_cfg.rotary)
            self.sin = self.sin.to(self.precision_cfg.rotary)
        # lm_head stays in compute precision but logits will be cast later.

    # -------------------------------------------------------------------------
    # Rotary embeddings

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(self.precision_cfg.rotary)
        sin = sin.to(self.precision_cfg.rotary)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    # -------------------------------------------------------------------------
    # Optimizer setup (identical to original GPT)

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        ddp, rank, _, _ = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(
            lm_head_params
        )
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    # -------------------------------------------------------------------------
    # Forward

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx on different devices: {idx.device} != {self.cos.device}"
        )
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]

        x = self.transformer.wte(idx)
        if x.dtype != self.precision_cfg.compute:
            x = x.to(self.precision_cfg.compute)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        if logits.dtype != self.precision_cfg.logits:
            logits = logits.to(self.precision_cfg.logits)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        return logits

    # -------------------------------------------------------------------------
    # Generation (unchanged except for dtype conversions)

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
