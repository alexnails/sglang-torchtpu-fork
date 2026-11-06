# SPDX-License-Identifier: Apache-2.0
"""Unit tests for speculative decoding draft KV caches over TpuUnifiedSWAKVPool."""

from __future__ import annotations

import types
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.model_config import AttentionArch, is_hybrid_swa_model
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_utils import get_dflash_attention_sliding_window_size
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    _kv_appender_specialization_key,
)
from sglang_tpu.dflash_draft_kv_write import (
    dflash_write_draft_kv,
    dflash_write_draft_kv_ref,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.speculative.block_draft_worker import TpuBlockDraftWorker
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.kv_free_fence import install_kv_free_fence
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWAPoolConfigurator,
    TpuUnifiedSWATokenToKVPoolAllocator,
    resolve_unified_draft_kv_pricing,
    unified_kv_pool_enabled,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_neox_rotary(head_dim: int, max_pos: int = 256) -> nn.Module:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    class _Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_neox_style = True
            self.rotary_dim = head_dim
            self.register_buffer("cos_sin_cache", cos_sin_cache)

        def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
            rows = self.cos_sin_cache.index_select(0, positions.to(torch.int64)).to(
                k.dtype
            )
            cos, sin = rows.chunk(2, dim=-1)
            k_shaped = k.view(-1, k.shape[-1] // head_dim, head_dim)
            x1, x2 = k_shaped.chunk(2, dim=-1)
            cos_b = cos.unsqueeze(1)
            sin_b = sin.unsqueeze(1)
            rot = torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)
            return q, rot.view_as(k)

    return _Rotary()


class _StubDraftAttn(nn.Module):
    def __init__(
        self,
        *,
        layer_id: int,
        hidden: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_size = num_kv_heads * head_dim
        self.rotary_emb = _make_neox_rotary(head_dim)
        self.kv_weight = nn.Parameter(
            torch.randn(2 * self.kv_size, hidden, dtype=dtype) / hidden**0.5
        )
        self.k_norm_weight = nn.Parameter(torch.rand(head_dim, dtype=dtype) + 0.5)

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv = F.linear(hidden_states, self.kv_weight)
        return kv.split([self.kv_size, self.kv_size], dim=-1)

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        by_head = k.reshape(-1, self.head_dim).float()
        normed = by_head * torch.rsqrt(by_head.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (normed.to(k.dtype) * self.k_norm_weight).view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        _, k_rot = self.rotary_emb(positions, k, k)
        return k_rot


class _StubDFlashModel(DFlashDraftModel):
    """Minimal DFlashDraftModel subclass bypassing heavy HF init for CPU unit tests."""

    def __init__(
        self,
        layer_specs: list[tuple[int, int, int]],
        hidden: int = 32,
        target_hidden: int = 48,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        nn.Module.__init__(self)
        self.fc = nn.Linear(target_hidden, hidden, bias=False, dtype=dtype)
        self.layers = nn.ModuleList()
        for layer_id, num_kv_heads, head_dim in layer_specs:
            layer = nn.Module()
            layer.self_attn = _StubDraftAttn(
                layer_id=layer_id,
                hidden=hidden,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            self.layers.append(layer)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(target_hidden)

    def prepare_context_hidden_for_kv(
        self, layer: nn.Module, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        del layer
        return ctx_hidden


def _pytorch_fused_rope_write_kv(
    *,
    dst_starts: torch.Tensor,
    src_starts: torch.Tensor,
    span_lens: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_bufs: list[torch.Tensor],
    max_span_len: int,
) -> None:
    """CPU reference of sglang_tpu.kv_fused_rope_write.fused_rope_write_kv."""
    del max_span_len
    num_layers, _, num_kv_heads, head_dim = k.shape
    x1, x2 = k.chunk(2, dim=-1)
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    k_rot = torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)
    for span_idx in range(int(span_lens.shape[0])):
        n = int(span_lens[span_idx].item())
        if n <= 0:
            continue
        src = int(src_starts[span_idx].item())
        dst = int(dst_starts[span_idx].item())
        loc = torch.arange(dst, dst + n, dtype=torch.long, device=k.device)
        for layer_idx in range(num_layers):
            buf = kv_bufs[layer_idx]
            kv_pair = torch.stack(
                [
                    k_rot[layer_idx, src : src + n],
                    v[layer_idx, src : src + n],
                ],
                dim=2,
            ).reshape(n, num_kv_heads * 2, head_dim)
            padded_head_dim = buf.shape[-1]
            num_kv_heads_x2 = buf.shape[2] * buf.shape[3]
            kv_pair = F.pad(
                kv_pair,
                (
                    0,
                    padded_head_dim - head_dim,
                    0,
                    num_kv_heads_x2 - num_kv_heads * 2,
                ),
            )
            buf.flatten(0, 1).index_copy_(
                0,
                loc,
                kv_pair.view(n, *buf.shape[2:]).to(buf.dtype),
            )


class TestDraftModelConfigAndCoPricing(unittest.TestCase):
    """Tests for draft ModelConfig classification and closed-form HBM co-pricing."""

    @contextmanager
    def _ctx(self, **overrides):
        defaults = dict(
            attention_backend="pallas_tpu_rpa_v3",
            tp_size=1,
            pp_size=1,
            dcp_size=1,
            attn_cp_size=1,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            hicache_io_backend="raiden",
            enable_unified_cache_external_linker=False,
            enable_unified_memory=False,
            page_size=16,
            disable_radix_cache=False,
            swa_full_tokens_ratio=0.5,
            chunked_prefill_size=512,
            disable_overlap_schedule=False,
            max_running_requests=8,
            speculative_algorithm=None,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
        )
        defaults.update(overrides)
        try:
            with ExitStack() as stack:
                stack.enter_context(tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.override(True))
                stack.enter_context(get_context().override_server_args(**defaults))
                stack.enter_context(
                    get_parallel().override(
                        attn_tp_size=1,
                        attn_dcp_size=1,
                        dcp_enabled=False,
                    )
                )
                yield
        finally:
            tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.clear()

    def _make_mock_mc(
        self,
        *,
        full_ids: list[int],
        swa_ids: list[int],
        head_num: int = 2,
        head_dim: int = 128,
        swa_head_num: int | None = None,
        swa_head_dim: int | None = None,
        sliding_window: int | None = 128,
        attention_arch=AttentionArch.MHA,
        has_asymmetric_kv: bool = False,
        mtp_local_layer_ids=None,
    ):
        swa_head_num = head_num if swa_head_num is None else swa_head_num
        swa_head_dim = head_dim if swa_head_dim is None else swa_head_dim
        is_swa = len(swa_ids) > 0
        hf_text_cfg = types.SimpleNamespace(
            architectures=["DummyForCausalLM"],
            sliding_window=sliding_window if is_swa else None,
            mtp_local_layer_ids=mtp_local_layer_ids,
        )
        hf_text_cfg.get_text_config = lambda: hf_text_cfg
        return types.SimpleNamespace(
            dtype=torch.bfloat16,
            attention_arch=attention_arch,
            has_asymmetric_kv=has_asymmetric_kv,
            is_hybrid_swa=is_swa,
            linear_attn_registry_result=None,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            num_hidden_layers=len(full_ids) + len(swa_ids),
            full_attention_layer_ids=list(full_ids),
            swa_attention_layer_ids=list(swa_ids),
            head_dim=head_dim,
            v_head_dim=head_dim,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_head_dim,
            sliding_window_size=sliding_window if is_swa else None,
            hf_config=hf_text_cfg,
            hf_text_config=hf_text_cfg,
            get_num_kv_heads=lambda tp=1, dcp=1: head_num,
            get_swa_num_kv_heads=lambda tp=1, dcp=1: swa_head_num,
            get_hybrid_layer_ids=lambda start, end: (list(full_ids), list(swa_ids)),
        )

    def _make_target_kvc(
        self,
        *,
        target_mc,
        spec_algorithm: SpeculativeAlgorithm,
        draft_mc=None,
        num_runners: int = 1,
        page_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        sliding_window_size: int = 127,
    ):
        full_ids, swa_ids = target_mc.get_hybrid_layer_ids(
            0, target_mc.num_hidden_layers
        )
        spec_aux = None
        if draft_mc is not None:
            total_draft_layers = num_runners * draft_mc.num_hidden_layers
            total_draft_swa = num_runners * len(draft_mc.swa_attention_layer_ids)
            spec_aux = types.SimpleNamespace(
                draft_model_config=draft_mc,
                eagle_draft_num_layers=total_draft_layers,
                eagle_draft_swa_num_layers=total_draft_swa,
                dflash_draft_num_layers=total_draft_layers,
            )
        return types.SimpleNamespace(
            device="cpu",
            server_args=types.SimpleNamespace(
                speculative_draft_model_path="dummy-draft",
                speculative_draft_model_revision=None,
            ),
            is_hybrid_swa=True,
            use_mla_backend=False,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            is_draft_worker=False,
            spec_algorithm=spec_algorithm,
            spec_aux_config=spec_aux,
            pp_size=1,
            attn_dp_size=1,
            _tpu_mha_pool_class=lambda: TpuFusedMHATokenToKVPool,
            kv_cache_dtype_str="bf16" if dtype == torch.bfloat16 else "fp8_e4m3",
            kv_cache_dtype=dtype,
            kv_group_size=None,
            pool_page_size=page_size,
            page_size=page_size,
            sliding_window_size=sliding_window_size,
            model_config=target_mc,
            layer_info=types.SimpleNamespace(
                start_layer=0,
                end_layer=target_mc.num_hidden_layers,
                full_attention_layer_ids=full_ids,
                swa_attention_layer_ids=swa_ids,
            ),
        )

    def test_draft_hybrid_swa_detection_and_dflash_window(self):
        fa_only_txt = types.SimpleNamespace(
            layer_types=["full_attention"] * 4,
            num_hidden_layers=4,
            sliding_window=128,
        )
        self.assertFalse(
            is_hybrid_swa_model(
                ["Gemma4ForCausalLM"],
                fa_only_txt,
                is_draft_model=True,
            )
        )
        self.assertTrue(
            is_hybrid_swa_model(
                ["Gemma4ForCausalLM"],
                types.SimpleNamespace(sliding_window=128),
                is_draft_model=False,
            )
        )
        for ltypes in (
            ["sliding_attention"] * 4,
            ["sliding_attention", "full_attention"] * 2,
        ):
            self.assertTrue(
                is_hybrid_swa_model(
                    ["LlamaForCausalLM"],
                    types.SimpleNamespace(
                        layer_types=ltypes, num_hidden_layers=4, sliding_window=128
                    ),
                    is_draft_model=True,
                )
            )
        dflash_cfg = {
            "layer_types": ["sliding_attention", "full_attention"],
            "dflash_config": {"swa_window_size": 256},
        }
        self.assertEqual(get_dflash_attention_sliding_window_size(dflash_cfg), 255)

    def test_closed_form_copricing_across_arbitrary_drafters(self):
        target_mc = self._make_mock_mc(
            full_ids=[0, 1],
            swa_ids=[2, 3, 4, 5, 6, 7],
            head_num=2,
            head_dim=128,
            sliding_window=128,
        )
        budget = 64 * 1024 * 1024  # 64 MB

        cases = [
            # (label, algo, d_full, d_swa, d_h, d_d, d_sh, d_sd, d_win, mtp_local, full_cap, num_runners)
            ("FA_only_eagle", SpeculativeAlgorithm.EAGLE, [0, 1, 2, 3], [], 1, 64, 1, 64, None, None, False, 1),
            ("SWA_only_compact_dflash", SpeculativeAlgorithm.DFLASH, [], [0, 1, 2, 3], 1, 64, 1, 64, 128, None, False, 1),
            ("Mixed_hetero_compact_dspark", SpeculativeAlgorithm.DSPARK, [1, 3], [0, 2, 4], 2, 64, 1, 128, 128, None, False, 1),
            ("Mixed_wider_window_standalone", SpeculativeAlgorithm.STANDALONE, [0], [1, 2], 1, 128, 4, 64, 512, None, True, 1),
            ("Multi_runner_MTP_compact", SpeculativeAlgorithm.EAGLE, [0], [1], 1, 64, 1, 64, 128, None, False, 2),
            ("MTP_local_layer_ids_full_cap", SpeculativeAlgorithm.EAGLE, [0], [1], 1, 64, 1, 64, 128, [1], True, 1),
        ]

        for (
            label,
            algo,
            d_full,
            d_swa,
            d_h,
            d_d,
            d_sh,
            d_sd,
            d_win,
            mtp_local,
            full_cap,
            num_runners,
        ) in cases:
            with self.subTest(label=label), self._ctx(speculative_algorithm=algo.name):
                draft_mc = self._make_mock_mc(
                    full_ids=d_full,
                    swa_ids=d_swa,
                    head_num=d_h,
                    head_dim=d_d,
                    swa_head_num=d_sh,
                    swa_head_dim=d_sd,
                    sliding_window=d_win,
                    mtp_local_layer_ids=mtp_local,
                )
                kvc = self._make_target_kvc(
                    target_mc=target_mc,
                    spec_algorithm=algo,
                    draft_mc=draft_mc,
                    num_runners=num_runners,
                    page_size=16,
                    sliding_window_size=127,  # window_left for sliding_window=128!
                )
                pricing = resolve_unified_draft_kv_pricing(kvc)
                if d_swa and not full_cap:
                    self.assertGreater(pricing.swa_compact_per_token, 0)
                else:
                    self.assertEqual(pricing.swa_compact_per_token, 0)
                    self.assertGreater(pricing.full_per_token, 0)

                conf = TpuUnifiedSWAPoolConfigurator(kvc)
                if d_swa and not full_cap:
                    self.assertEqual(conf._align_pages, 24)
                else:
                    self.assertEqual(conf._align_pages, 8)

                sized = conf.calculate_pool_sizes(budget, page_size=16)
                n_tokens = sized.full_max_total_num_tokens
                n_pages = n_tokens // 16
                self.assertEqual(n_pages % conf._align_pages, 0)

                t_pool = TpuUnifiedSWAKVPool(
                    size=n_tokens,
                    page_size=16,
                    dtype=torch.bfloat16,
                    full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=[0, 1]),
                    swa=AttnLayerSpec(
                        head_num=2, head_dim=128, layer_ids=[2, 3, 4, 5, 6, 7]
                    ),
                    device="cpu",
                )
                total_bytes = sum(t_pool.get_kv_size_bytes())
                if d_swa:
                    d_swa_tokens = n_tokens if full_cap else (n_pages // 3) * 16
                    self.assertEqual((d_swa_tokens // 16) % 8, 0)
                    d_pool = SWAKVPool(
                        size=n_tokens,
                        size_swa=d_swa_tokens,
                        page_size=16,
                        dtype=torch.bfloat16,
                        head_num=d_h,
                        head_dim=d_d,
                        swa_attention_layer_ids=d_swa,
                        full_attention_layer_ids=d_full,
                        device="cpu",
                        full_kv_pool_class=TpuFusedMHATokenToKVPool,
                        swa_kv_pool_class=TpuFusedMHATokenToKVPool,
                        full_kv_pool_kwargs={
                            "page_size": 16,
                            "enable_memory_saver": False,
                            "device": "cpu",
                            "head_num": d_h,
                            "head_dim": d_d,
                            "v_head_dim": d_d,
                        },
                        swa_kv_pool_kwargs={
                            "page_size": 16,
                            "enable_memory_saver": False,
                            "device": "cpu",
                            "head_num": d_sh,
                            "head_dim": d_sd,
                            "v_head_dim": d_sd,
                        },
                    )
                else:
                    d_pool = TpuFusedMHATokenToKVPool(
                        size=n_tokens,
                        page_size=16,
                        dtype=torch.bfloat16,
                        head_num=d_h,
                        head_dim=d_d,
                        layer_num=len(d_full),
                        device="cpu",
                        enable_memory_saver=False,
                    )
                total_bytes += num_runners * sum(d_pool.get_kv_size_bytes())
                self.assertLessEqual(total_bytes, budget)

    def test_unsupported_mla_or_asymmetric_draft_rejected(self):
        target_mc = self._make_mock_mc(
            full_ids=[0, 1], swa_ids=[2, 3], head_num=2, head_dim=128
        )
        mla_draft = self._make_mock_mc(
            full_ids=[0, 1],
            swa_ids=[],
            attention_arch=AttentionArch.MLA,
        )
        asym_draft = self._make_mock_mc(
            full_ids=[0, 1],
            swa_ids=[],
            has_asymmetric_kv=True,
        )
        with self._ctx(speculative_algorithm="DFLASH"):
            for bad_draft in (mla_draft, asym_draft):
                kvc = self._make_target_kvc(
                    target_mc=target_mc,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                    draft_mc=bad_draft,
                )
                with self.assertRaisesRegex(ValueError, "--enable-unified-memory"):
                    unified_kv_pool_enabled(kvc)


class TestDraftSwaSidecarAndFence(unittest.TestCase):
    """Tests for _DraftSwaSidecar page lockstep, slot translation, and KVFreeFence safety."""

    def _build_target_and_draft(
        self,
        *,
        num_target_pages: int = 24,
        page_size: int = 8,
        m_swa: int = 3,
        draft_full_ids: list[int] = [0],
        draft_swa_ids: list[int] = [1],
        full_capacity: bool = False,
    ):
        size = num_target_pages * page_size
        t_pool = TpuUnifiedSWAKVPool(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=[0, 1]),
            swa=AttnLayerSpec(
                head_num=1,
                head_dim=64,
                layer_ids=list(range(2, 2 + 2 * m_swa)),
            ),
            device="cpu",
        )
        req_pool = ReqToTokenPool(
            size=8, max_context_len=256, device="cpu", enable_memory_saver=False
        )
        t_alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=t_pool,
            need_sort=False,
            req_to_token_pool=req_pool,
            sliding_window_size=16,
        )
        d_kvc = types.SimpleNamespace(
            is_hybrid_swa=len(draft_swa_ids) > 0,
            sliding_window_size=128 if full_capacity else 8,
            draft_swa_full_capacity=False,
            layer_info=types.SimpleNamespace(
                full_attention_layer_ids=list(draft_full_ids),
                swa_attention_layer_ids=list(draft_swa_ids),
            ),
            model_config=types.SimpleNamespace(sliding_window_size=None),
        )
        max_tok, full_tok, swa_tok = t_alloc.derive_draft_pool_sizes(
            d_kvc,
            max_total_num_tokens=0,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        if draft_swa_ids:
            d_pool = SWAKVPool(
                size=full_tok,
                size_swa=swa_tok,
                page_size=page_size,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=64,
                swa_attention_layer_ids=draft_swa_ids,
                full_attention_layer_ids=draft_full_ids,
                device="cpu",
                full_kv_pool_class=TpuFusedMHATokenToKVPool,
                swa_kv_pool_class=TpuFusedMHATokenToKVPool,
                full_kv_pool_kwargs={
                    "page_size": page_size,
                    "enable_memory_saver": False,
                    "device": "cpu",
                    "head_num": 1,
                    "head_dim": 64,
                    "v_head_dim": 64,
                },
                swa_kv_pool_kwargs={
                    "page_size": page_size,
                    "enable_memory_saver": False,
                    "device": "cpu",
                    "head_num": 2,
                    "head_dim": 128,
                    "v_head_dim": 128,
                },
            )
        else:
            d_pool = TpuFusedMHATokenToKVPool(
                size=max_tok,
                page_size=page_size,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=64,
                layer_num=len(draft_full_ids),
                device="cpu",
                enable_memory_saver=False,
            )
        t_alloc.attach_draft_kv_pool(d_kvc, d_pool)
        return t_pool, t_alloc, d_pool, req_pool

    def test_sidecar_extend_decode_evict_and_fence_invariants(self):
        # 24 target pages, page_size=8, M_T=3 -> 8 draft SWA pages (64 compact SWA tokens).
        _, t_alloc, d_pool, req_pool = self._build_target_and_draft(
            num_target_pages=24, page_size=8, m_swa=3
        )
        sidecar = t_alloc.draft_swa_sidecar
        self.assertIsNotNone(sidecar)
        self.assertEqual(d_pool.size, 192)
        self.assertEqual(d_pool.size_swa, 64)
        self.assertIs(d_pool.full_to_swa_index_mapping, sidecar.mapping)
        self.assertEqual(int(d_pool.full_to_swa_index_mapping[0].item()), 0)
        self.assertEqual(int(d_pool.full_to_swa_index_mapping[-1].item()), -1)

        fence = install_kv_free_fence(
            t_alloc,
            synchronize_device=lambda: None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            copies_draft_tail=True,
        )
        fence.attach(sidecar.allocator)  # idempotent re-attach!
        self.assertEqual(len(fence._allocators), 3)

        result = types.SimpleNamespace(copy_done=None)
        fence.on_batch_formed()
        fence.on_forward_launched(result)
        with fence.forward_write_target():
            # Extend 20 tokens (3 logical pages -> consumes 3 + 3*3 = 12 target pages, 3 draft SWA pages).
            ext = t_alloc.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([20], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([20], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64),
                extend_num_tokens=20,
            )
            self.assertIsNotNone(ext)
            req_pool.req_to_token[0, :20] = ext
            self.assertEqual(sidecar.allocator.available_size() // 8, 5)

            # Verify all 20 tokens map to distinct valid slots in [8, 72).
            d_swa_slots = d_pool.translate_loc_from_full_to_swa(ext)
            self.assertEqual(len(set(d_swa_slots.tolist())), 20)
            self.assertTrue(torch.all((d_swa_slots >= 8) & (d_swa_slots < 72)))
            self.assertEqual(
                int(
                    d_pool.translate_loc_from_full_to_swa(torch.tensor([-1])).item()
                ),
                -1,
            )

            # Evict first 2 SWA pages (tokens 0..15) via free_swa; fence holds them!
            t_alloc.free_swa(ext[:16])
            self.assertTrue(torch.all(d_pool.full_to_swa_index_mapping[ext[:16]] == 0))

        self.assertEqual(sidecar.allocator.free_pages.numel(), 5)
        self.assertEqual(sidecar.allocator.held_pages.num_pages, 2)
        self.assertEqual(sidecar.allocator.available_size() // 8, 7)

        # Non-forward allocation (simulating DMA reload while fence holds pages) never starves!
        ext2 = t_alloc.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([16], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=16,
        )
        self.assertIsNotNone(ext2)
        self.assertEqual(fence.stats.device_syncs, 0)
        self.assertEqual(sidecar.allocator.free_pages.numel(), 3)
        self.assertEqual(sidecar.allocator.held_pages.num_pages, 2)
        self.assertEqual(sidecar.allocator.available_size() // 8, 5)

        # Releasing the fence returns the 2 held draft SWA pages cleanly.
        fence.on_forward_done(result)
        self.assertEqual(sidecar.allocator.held_pages.num_pages, 0)
        self.assertEqual(sidecar.allocator.free_pages.numel(), 5)
        self.assertEqual(sidecar.allocator.available_size() // 8, 5)

        # Decode across a page boundary (pos 20..23 stay on page 3; pos 24 allocates new page).
        for pos in (20, 21, 22, 23, 24):
            prev_last = req_pool.req_to_token[0, pos - 1 : pos]
            seq_t = torch.tensor([pos + 1], dtype=torch.int64)
            dec = t_alloc.alloc_decode(seq_t, seq_t, prev_last)
            self.assertIsNotNone(dec)
            req_pool.req_to_token[0, pos] = dec[0]
            mapped_dec = int(d_pool.translate_loc_from_full_to_swa(dec)[0].item())
            self.assertGreaterEqual(mapped_dec, 8)
            self.assertLess(mapped_dec, 72)

        # Free everything -> all 8 draft SWA pages and 24 target pages restored!
        t_alloc.free(req_pool.req_to_token[0, :25])
        t_alloc.free(ext2)
        self.assertEqual(sidecar.allocator.available_size() // 8, 8)
        self.assertEqual(t_alloc.full_available_size(), 192)

    def test_wider_draft_window_uses_cpu_identity_mapping(self):
        _, t_alloc, d_pool, _ = self._build_target_and_draft(
            num_target_pages=16, page_size=8, m_swa=2, full_capacity=True
        )
        self.assertIsNone(t_alloc.draft_swa_sidecar)
        self.assertEqual(d_pool.size, 128)
        self.assertEqual(d_pool.size_swa, 128)
        mapping = d_pool.full_to_swa_index_mapping
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.device.type, "cpu")
        self.assertEqual(int(mapping[-1].item()), -1)
        self.assertTrue(
            torch.equal(mapping[:-1], torch.arange(136, dtype=torch.int64))
        )

    def test_multi_head_mtp_shares_single_sidecar(self):
        _, t_alloc, d_pool1, _ = self._build_target_and_draft(
            num_target_pages=24, page_size=8, m_swa=3
        )
        d_kvc2 = types.SimpleNamespace(
            is_hybrid_swa=True,
            sliding_window_size=8,
            draft_swa_full_capacity=False,
            layer_info=types.SimpleNamespace(
                full_attention_layer_ids=[0],
                swa_attention_layer_ids=[1],
            ),
            model_config=types.SimpleNamespace(sliding_window_size=None),
        )
        _, full_tok, swa_tok = t_alloc.derive_draft_pool_sizes(
            d_kvc2,
            max_total_num_tokens=0,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        d_pool2 = SWAKVPool(
            size=full_tok,
            size_swa=swa_tok,
            page_size=8,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            device="cpu",
            full_kv_pool_class=TpuFusedMHATokenToKVPool,
            swa_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        t_alloc.attach_draft_kv_pool(d_kvc2, d_pool2)
        self.assertIs(
            d_pool1.full_to_swa_index_mapping,
            d_pool2.full_to_swa_index_mapping,
        )


class TestDFlashDualSubpoolWriterAndWorkers(unittest.TestCase):
    """Tests for zero-kernel-change dual sub-pool DFlash KV writing and compile/worker hooks."""

    def test_dflash_write_draft_kv_matches_ref_across_fa_swa_and_hetero_mixed(self):
        torch.manual_seed(7)
        page_size = 8
        max_reqs = 8

        configs = [
            ("fa_only", [(0, 2, 64), (1, 2, 64), (2, 2, 64)], [0, 1, 2], []),
            ("swa_only", [(0, 1, 128), (1, 1, 128), (2, 1, 128)], [], [0, 1, 2]),
            (
                "mixed_hetero",
                [(0, 1, 128), (1, 2, 64), (2, 1, 128), (3, 2, 64)],
                [1, 3],
                [0, 2],
            ),
        ]

        with patch(
            "sglang_tpu.kv_fused_rope_write.fused_rope_write_kv",
            side_effect=_pytorch_fused_rope_write_kv,
        ):
            for label, layer_specs, full_ids, swa_ids in configs:
                with self.subTest(label=label):
                    model = _StubDFlashModel(layer_specs, hidden=32, target_hidden=48)

                    def _make_pool():
                        if not swa_ids:
                            return TpuFusedMHATokenToKVPool(
                                size=64,
                                page_size=page_size,
                                dtype=torch.bfloat16,
                                head_num=2,
                                head_dim=64,
                                layer_num=len(full_ids),
                                device="cpu",
                                enable_memory_saver=False,
                            )
                        return SWAKVPool(
                            size=64,
                            size_swa=32,
                            page_size=page_size,
                            dtype=torch.bfloat16,
                            head_num=2,
                            head_dim=64,
                            swa_attention_layer_ids=swa_ids,
                            full_attention_layer_ids=full_ids,
                            device="cpu",
                            full_kv_pool_class=TpuFusedMHATokenToKVPool,
                            swa_kv_pool_class=TpuFusedMHATokenToKVPool,
                            full_kv_pool_kwargs={
                                "page_size": page_size,
                                "enable_memory_saver": False,
                                "device": "cpu",
                                "head_num": 2,
                                "head_dim": 64,
                                "v_head_dim": 64,
                            },
                            swa_kv_pool_kwargs={
                                "page_size": page_size,
                                "enable_memory_saver": False,
                                "device": "cpu",
                                "head_num": 1,
                                "head_dim": 128,
                                "v_head_dim": 128,
                            },
                        )

                    # --- Extend path ---
                    pool_fused = _make_pool()
                    pool_ref = _make_pool()
                    ctx_hidden = torch.randn(12, 32, dtype=torch.bfloat16)
                    positions = torch.arange(5, 17, dtype=torch.int64)
                    full_loc = torch.tensor(
                        [10, 11, 12, 13, 14, 15, 24, 25, 26, 40, 41, 42],
                        dtype=torch.int64,
                    )
                    swa_loc = torch.tensor(
                        [18, 19, 20, 21, 22, 23, 8, 9, 10, 24, 25, 26],
                        dtype=torch.int64,
                    )

                    loc_arg = swa_loc if (swa_ids and not full_ids) else full_loc
                    swa_arg = swa_loc if (swa_ids and full_ids) else None

                    dflash_write_draft_kv(
                        draft_model=model,
                        kv_pool=(
                            pool_fused.swa_kv_pool
                            if (swa_ids and not full_ids)
                            else pool_fused
                        ),
                        ctx_hidden=ctx_hidden,
                        ctx_positions=positions,
                        ctx_cache_loc=loc_arg,
                        cache_loc_2d=None,
                        commit_lens=None,
                        max_num_reqs=max_reqs,
                        swa_cache_loc=swa_arg,
                    )
                    dflash_write_draft_kv_ref(
                        draft_model=model,
                        kv_pool=(
                            pool_ref.swa_kv_pool
                            if (swa_ids and not full_ids)
                            else pool_ref
                        ),
                        ctx_hidden=ctx_hidden,
                        ctx_positions=positions,
                        ctx_cache_loc=loc_arg,
                        swa_cache_loc=swa_arg,
                    )
                    for b_f, b_r in zip(pool_fused.get_state(), pool_ref.get_state()):
                        torch.testing.assert_close(b_f, b_r, atol=1e-2, rtol=1e-2)

                    # --- Verify path (partial commits crossing page boundary) ---
                    pool_v_fused = _make_pool()
                    pool_v_ref = _make_pool()
                    v_hidden = torch.randn(8, 32, dtype=torch.bfloat16)
                    v_pos = torch.tensor([6, 7, 8, 9, 14, 15, 16, 17], dtype=torch.int64)
                    v_full_2d = torch.tensor(
                        [[14, 15, 24, 25], [33, 34, 35, 36]], dtype=torch.int64
                    )
                    v_swa_2d = torch.tensor(
                        [[14, 15, 8, 9], [17, 18, 19, 20]], dtype=torch.int64
                    )
                    commit_lens = torch.tensor([3, 2], dtype=torch.int32)

                    v_loc_2d = v_swa_2d if (swa_ids and not full_ids) else v_full_2d
                    v_swa_arg_2d = v_swa_2d if (swa_ids and full_ids) else None

                    dflash_write_draft_kv(
                        draft_model=model,
                        kv_pool=(
                            pool_v_fused.swa_kv_pool
                            if (swa_ids and not full_ids)
                            else pool_v_fused
                        ),
                        ctx_hidden=v_hidden,
                        ctx_positions=v_pos,
                        ctx_cache_loc=v_loc_2d.reshape(-1),
                        cache_loc_2d=v_loc_2d,
                        commit_lens=commit_lens,
                        max_num_reqs=max_reqs,
                        swa_cache_loc=(
                            v_swa_arg_2d.reshape(-1)
                            if v_swa_arg_2d is not None
                            else None
                        ),
                        swa_cache_loc_2d=v_swa_arg_2d,
                    )
                    keep = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long)
                    dflash_write_draft_kv_ref(
                        draft_model=model,
                        kv_pool=(
                            pool_v_ref.swa_kv_pool
                            if (swa_ids and not full_ids)
                            else pool_v_ref
                        ),
                        ctx_hidden=v_hidden[keep],
                        ctx_positions=v_pos[keep],
                        ctx_cache_loc=v_loc_2d.reshape(-1)[keep],
                        swa_cache_loc=(
                            v_swa_arg_2d.reshape(-1)[keep]
                            if v_swa_arg_2d is not None
                            else None
                        ),
                    )
                    for b_f, b_r in zip(
                        pool_v_fused.get_state(), pool_v_ref.get_state()
                    ):
                        torch.testing.assert_close(b_f, b_r, atol=1e-2, rtol=1e-2)

    def test_compile_policy_and_workers_dual_cache_loc_routing(self):
        base_kwargs = dict(
            target_hidden=torch.zeros(8, 16),
            cache_loc=torch.zeros(8, dtype=torch.int64),
            positions=torch.zeros(8, dtype=torch.int64),
            cache_loc_2d=torch.zeros(2, 4, dtype=torch.int64),
            commit_lens=torch.zeros(2, dtype=torch.int32),
        )
        k_single = _kv_appender_specialization_key(**base_kwargs)
        k_dual = _kv_appender_specialization_key(
            **base_kwargs,
            swa_cache_loc=torch.zeros(8, dtype=torch.int64),
            swa_cache_loc_2d=torch.zeros(2, 4, dtype=torch.int64),
        )
        self.assertNotEqual(k_single, k_dual)

        mixed_pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
            full_kv_pool_class=TpuFusedMHATokenToKVPool,
            swa_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        mapping = torch.zeros(37, dtype=torch.int64)
        mapping[-1] = -1
        mapping[8:12] = torch.tensor([16, 17, 18, 19])
        mixed_pool.register_mapping(mapping)

        captured_calls = []
        v2_worker = object.__new__(DFlashWorkerV2)
        v2_worker.draft_model_runner = types.SimpleNamespace(token_to_kv_pool=mixed_pool)
        v2_worker.append_target_hidden_to_draft_kv_by_loc_tpu = (
            lambda **kw: captured_calls.append(kw)
        )
        with patch("sglang.srt.speculative.dflash_worker_v2._is_tpu", True):
            v2_worker._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=torch.ones(4, 16),
                cache_loc=torch.tensor([8, 9, 10, 11], dtype=torch.int64),
                positions=torch.arange(4, dtype=torch.int64),
                cache_loc_2d=torch.tensor([[8, 9, 10, 11]], dtype=torch.int64),
                commit_lens=torch.tensor([3], dtype=torch.int32),
            )
        self.assertEqual(len(captured_calls), 1)
        self.assertTrue(
            torch.equal(
                captured_calls[0]["cache_loc"],
                torch.tensor([8, 9, 10, 11], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                captured_calls[0]["swa_cache_loc"],
                torch.tensor([16, 17, 18, 19], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                captured_calls[0]["swa_cache_loc_2d"],
                torch.tensor([[16, 17, 18, 19]], dtype=torch.int64),
            )
        )

        swa_only_pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[],
            device="cpu",
            full_kv_pool_class=TpuFusedMHATokenToKVPool,
            swa_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        swa_only_pool.register_mapping(mapping)
        captured_calls.clear()
        v2_worker.draft_model_runner.token_to_kv_pool = swa_only_pool
        with patch("sglang.srt.speculative.dflash_worker_v2._is_tpu", True):
            v2_worker._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=torch.ones(4, 16),
                cache_loc=torch.tensor([8, 9, 10, 11], dtype=torch.int64),
                positions=torch.arange(4, dtype=torch.int64),
            )
        self.assertEqual(len(captured_calls), 1)
        self.assertTrue(
            torch.equal(
                captured_calls[0]["cache_loc"],
                torch.tensor([16, 17, 18, 19], dtype=torch.int64),
            )
        )
        self.assertNotIn("swa_cache_loc", captured_calls[0])

    def test_block_draft_and_spec_workers_mixed_and_swa_only(self):
        mixed_pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
            full_kv_pool_class=TpuFusedMHATokenToKVPool,
            swa_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        mapping = torch.zeros(37, dtype=torch.int64)
        mapping[-1] = -1
        mapping[4:8] = torch.tensor([12, 13, 14, 15])
        mixed_pool.register_mapping(mapping)

        class _ConcreteBlockDraft(TpuBlockDraftWorker):
            def sample_drafts(self, **kw):
                raise NotImplementedError

        draft_worker = object.__new__(_ConcreteBlockDraft)
        draft_model = _StubDFlashModel([(0, 1, 64), (1, 1, 64)], hidden=16, target_hidden=16)
        draft_worker.model_runner = types.SimpleNamespace(
            token_to_kv_pool=mixed_pool,
            model=draft_model,
            req_to_token_pool=types.SimpleNamespace(size=4),
        )
        self.assertIs(draft_worker.full_to_swa, mapping)
        self.assertTrue(draft_worker.needs_dual_cache_loc)

        with patch(
            "sglang_tpu.kv_fused_rope_write.fused_rope_write_kv",
            side_effect=_pytorch_fused_rope_write_kv,
        ):
            tail = draft_worker._write_draft_kv_eager(
                target_hidden=torch.randn(4, 16, dtype=torch.bfloat16),
                cache_loc=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
                positions=torch.arange(4, dtype=torch.int64),
                swa_cache_loc=torch.tensor([12, 13, 14, 15], dtype=torch.int64),
            )
            self.assertEqual(tail.dtype, torch.float32)

        recorded = []
        draft_worker.draft_context = ExitStack
        draft_worker.write_draft_kv = lambda **kw: recorded.append(kw) or torch.zeros(1)
        spec_worker = object.__new__(TpuBlockSpecWorker)
        spec_worker._block_draft_worker = draft_worker
        spec_worker.device = torch.device("cpu")
        batch = types.SimpleNamespace(
            prefix_lens=[0],
            extend_lens=[4],
            out_cache_loc=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
        )
        spec_worker.extend_draft_from_hidden(
            batch,
            target_hidden=torch.zeros(8, 16),
            cache_loc=torch.tensor([4, 5, 6, 7, 0, 0, 0, 0], dtype=torch.int64),
            bonus_tokens=torch.zeros(1, dtype=torch.int64),
        )
        self.assertEqual(len(recorded), 1)
        self.assertTrue(
            torch.equal(
                recorded[0]["cache_loc"],
                torch.tensor([4, 5, 6, 7, 0, 0, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                recorded[0]["swa_cache_loc"],
                torch.tensor([12, 13, 14, 15, 0, 0, 0, 0], dtype=torch.int64),
            )
        )


if __name__ == "__main__":
    unittest.main()
