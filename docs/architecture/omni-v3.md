# Nemotron-3-Nano-Omni-30B-A3B V3 Architecture Research

**Target model:** `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
**Architecture string:** `NemotronH_Nano_Omni_Reasoning_V3`
**Research date:** 2026-05-11
**TRT-LLM version being studied:** v1.2.1 (user's current install)
**TRT-LLM version with V3 support:** v1.3.0rc13+ (released 2026-04-29)

---

## 1. Model Overview

### Release context

NVIDIA released Nemotron 3 Nano Omni on **April 28, 2026**, simultaneously on Hugging Face, build.nvidia.com, and NGC. This is the first model in the Nemotron family with native audio support (hence "Omni" — text + image + video + audio in one model).

**arXiv paper:** 2604.24954 — "Nemotron 3 Nano Omni" (full PDF; HTML abstract only publicly accessible)

**Official blogs:**
- https://blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents/
- https://developer.nvidia.com/blog/nvidia-nemotron-3-nano-omni-powers-multimodal-agent-reasoning-in-a-single-efficient-open-model/
- https://huggingface.co/blog/nvidia/nemotron-3-nano-omni-multimodal-intelligence

### What changed V2 → V3

| Aspect | V2 (NemotronH_Nano_VL_V2) | V3 (NemotronH_Nano_Omni_Reasoning_V3) |
|--------|--------------------------|---------------------------------------|
| Backbone | Dense Nemotron Nano 12B hybrid | MoE Nemotron Nano 30B-A3B hybrid |
| MoE type | Flat (no grouped routing) | DeepSeek-V3-style grouped (128 experts, n_groups=8, topk_group=1) |
| Audio | None | Parakeet-TDT-0.6B-v2 + SoundEncoder |
| Video | RADIO + Conv3D | RADIO + Conv3D + tubelet embedding (video_temporal_patch_size=2) |
| Context | 128k tokens | 256k tokens |
| Reasoning | No | Yes (`nano-v3` reasoning parser) |
| Dynamic resolution | Fixed tiling | `vision_config.args.min_num_patches` based dynamic tiler |

### Scale
- 30B total / ~3B active per forward pass
- 256k token context window; 65,536 max output tokens
- NVFP4: 20.9 GB (routed experts FP4 E2M1 + FP8 scales; encoders BF16)
- Training set: 354M samples / 717B tokens across 1,395 datasets

---

## 2. LLM Architecture (NemotronH backbone)

### Hybrid layer structure

Per the HuggingFace blog and Nemotron 3 technical paper (arXiv 2512.20856):

- **23 Mamba-2 layers** — selective state-space for long-context memory efficiency
- **23 MoE layers** — 128 routed experts + 1 shared expert, 6 active per token
- **6 grouped-query attention layers** — for global interaction

Total: 52 layers. Mamba-2 and MoE layers interleave; attention layers are sparse (approximately every ~8 layers). The exact per-layer pattern is in the model's `config.json` under `llm_config.hybrid_override_pattern` (a string like "M*T*A...").

### V3 MoE routing math (DeepSeek-V3 grouped routing)

The TRT-LLM handler at `tensorrt_llm/_torch/models/modeling_nemotron_h.py` uses `DeepseekV3Gate` directly. This is confirmed DeepSeek-V3-style routing — not a NVIDIA-specific variant.

**Config params for V3:**
```
n_routed_experts    = 128
num_experts_per_tok = 6   (top_k)
n_groups            = 8   (groups of 128/8 = 16 experts each)
topk_group          = 1   (select 1 group per token)
routed_scaling_factor = 2.5
n_shared_experts    = 1
moe_intermediate_size = 1856
moe_shared_expert_intermediate_size = 3712
```

**Routing algorithm** (implemented in `Deepseekv3RoutingImpl` inside TRT-LLM):

1. Compute per-expert logits via linear projection of hidden state.
2. Apply sigmoid (not softmax) to get per-expert scores.
3. Add `e_score_correction_bias` (learned per-expert bias for load balancing without auxiliary loss — from DeepSeek's "aux-loss-free" paper arXiv 2408.15664).
4. Group the 128 experts into `n_groups=8` groups of 16 experts each.
5. For each group, compute the group score = sum of top-2 expert scores within the group.
6. Select `topk_group=1` group with highest group score.
7. Within that selected group, pick top-6 experts (but since `topk_group=1` and group has 16, we pick 6 out of 16).
8. Re-normalize selected expert scores by dividing by their sum, then multiply by `routed_scaling_factor=2.5`.
9. Route token to selected experts; output = weighted sum of expert outputs + shared expert output.

**LatentMoE flag:** V3 does NOT use LatentMoE (that is Nemotron Super/Ultra). V3's `moe_latent_size` is not set (None), so `use_latent_moe=False` and `moe_hidden_size = hidden_size`.

**Shared expert:** `moe_shared_expert_intermediate_size=3712` × `n_shared_experts=1` = one MLP with intermediate=3712 that always runs unconditionally, then added to the routed expert sum.

### Why V1.2.1 fails on V3

In TRT-LLM v1.2.1, the `NemotronH_Nano_Omni_Reasoning_V3` string is **not registered** in `register_auto_model`. The handler only knows `NemotronH_Nano_VL_V2`. Even if you alias it, the V2 MoE handler expected a **flat** (non-grouped) MoE config; the `n_group` and `topk_group` fields in the V3 config would hit shape mismatches:
- V2 used `intermediate_size` for the MoE dim; V3 uses separate `moe_intermediate_size=1856`
- V2's `NemotronHMOE` didn't instantiate `DeepseekV3Gate` — it used standard top-k without groups

The 2688-vs-672 shape mismatch the user observed comes from the MoE expert weight shape: V2 flat experts had shape `[num_experts, 4×hidden/tp]` while V3 grouped MoE expert intermediate is 1856 (not 672 = hidden/tp). The factor of 4 mismatch (2688/672 = 4) suggests V2 tried to apply its dense-MLP intermediate formula to a MoE weight that's sized for a much smaller intermediate.

---

## 3. Vision Encoder (C-RADIOv4-H)

### RADIO model identity

The vision encoder is **C-RADIOv4-H** — the Huge (653M parameter) variant of Compressed RADIO v4. This is a ViT-H class Vision Transformer with 16-pixel patch size, up to 2048×2048 input.

Model card: https://huggingface.co/nvidia/C-RADIOv4-H

RADIO outputs two tensors per image: `summary` (CLS-equivalent, shape `[B, C]`) and `features` (spatial patches, shape `[B, T, D]`). The Omni model uses the spatial `features` path.

### How images flow through V3

1. Image → dynamic resolution tiler (`DynamicResolutionImageTiler`): scales image preserving aspect ratio to fit within a token budget between `min_num_patches` (1024) and `max_num_patches` (13312). For square images this is 512×512 to ~1840×1840.
2. Pixel shuffle downsample 2× (ratio=0.5): groups 2×2 patch grids → 1 token; halves token count.
3. RMSNorm + Linear(vit_hidden×4 → projector_hidden) + SquaredReLU + Linear(projector_hidden → llm_hidden): the `mlp1` 2-layer projector.
4. Visual tokens replace `<image>` placeholders in the LLM input sequence.

**Implementation class:** `NanoV2VLVisionEncoder` in `tensorrt_llm/_torch/models/modeling_nemotron_nano.py`
**RADIO TRT-LLM handler:** `tensorrt_llm/_torch/models/modeling_radio.py`

### V3-specific video_embedder (tubelet embedding)

V3 introduces `video_temporal_patch_size=2` in `vision_config`, which activates the **tubelet embedding path** in RADIO. Two consecutive video frames are fused into a single "tubelet" before entering the ViT, halving temporal tokens. This is implemented via a Conv3D layer in RADIO (`model.patch_generator.video_embedder.weight` in the checkpoint — a 3D convolutional weight not present in V2's RADIO checkpoint).

V2 had `video_temporal_patch_size=1` (each frame processed independently). This is why the V3 checkpoint has the `video_embedder` tensor that causes shape mismatches when loading with a V2 handler: V2's RADIO has a 2D patch embedding while V3's RADIO has a 3D Conv patch embedding.

**EVS (Efficient Video Sampling):** After RADIO extracts features, static tokens (scene areas that didn't change between frames) are pruned. Only "dynamic" tokens are kept. The pruning rate is controlled by `--video-pruning-rate` at serve time (default 0.5 = remove 50% static tokens from non-first frames).

### TRT-LLM RADIO handler

`tensorrt_llm/_torch/models/modeling_radio.py` — supports both 2D and 3D paths via `num_frames` arg:
```python
vit_embeds = self.vision_model(micro_batch_pixel_values, num_frames=num_frames)
```
When `num_frames` is passed, RADIO uses the Conv3D tubelet path. When omitted, standard 2D path. This was added in TRT-LLM v1.3.0rc12 ("Add video temporal compression for Nemotron Nano and RADIO").

---

## 4. Audio Encoder (Parakeet-TDT)

### Parakeet model identity

**Model:** `nvidia/parakeet-tdt-0.6b-v2`
**Architecture:** FastConformer-XL (600M parameters) encoder + TDT (Transducer with Duration Terms) decoder
**Training:** Initialized from wav2vec SSL checkpoint, fine-tuned on 120,000 hours of English speech

In the Omni model, only the **encoder** is used (the TDT decoder is discarded). The encoder outputs continuous embeddings which are projected into the LLM's embedding space.

### Audio feature extraction pipeline

1. Resample to **16 kHz mono**
2. Compute **log-mel spectrogram** (hop size 10ms → 100 frames/second)
3. **3× stride-2 convolutional subsampling** in the FastConformer encoder → ~8× temporal downsampling → **~12.5 audio tokens/second**
4. FastConformer XL self-attention layers process the subsampled sequence
5. **Long audio chunking:** Audio is split into 30-second clips with 0.1s minimum. The `ParakeetExtractor` in TRT-LLM handles this splitting.

### MLP projector (SoundProjection)

`ProjectedParakeet` in `tensorrt_llm/_torch/models/modeling_parakeet.py`:
```python
class ParakeetProjection(nn.Module):
    # RMSNorm → Linear(hidden → projection_hidden) → SquaredReLU → Linear(projection_hidden → llm_hidden)
```
This is the same architecture as the vision projector (2-layer MLP with RMSNorm + SquaredReLU).

### Sound token flow in the LLM

1. Audio → `ParakeetExtractor` (mel spectrogram) → `ProjectedParakeet.encoder` (FastConformer) → `ProjectedParakeet.projection` (MLP)
2. Output embeddings have shape `[num_audio_tokens, llm_hidden_size]` where `num_audio_tokens ≈ audio_seconds × 12.5`
3. Audio tokens are inserted at `<so_embedding>` placeholder positions in the LLM token sequence, between `<so_start>` and `<so_end>` boundary tokens
4. `sound_context_token_id` config field marks which positions to replace

**Key config fields from `sound_config`:**
```
num_mel_bins             = 80 (standard mel spectrogram bins)
sampling_rate            = 16000
subsampling_factor       = 8
hidden_size              = FastConformer hidden dim (not published, ~1024 for XL)
projection_hidden_size   = MLP intermediate dim
```

### TRT-LLM implementation status

`modeling_parakeet.py` was added in TRT-LLM v1.3.0rc13 (April 29, 2026) as part of the Omni V3 support. It wraps `transformers.ParakeetEncoder` (HuggingFace) directly — meaning the audio encoder runs as eager PyTorch, not compiled TRT-LLM kernels. This is the same approach used for the vision encoder.

---

## 5. TRT-LLM Port Plan

### Critical finding: V3 is already supported in v1.3.0rc13+

The user is on **v1.2.1** (released 2026-04-20). V3 support landed in **v1.3.0rc13** (released 2026-04-29), nine days later.

**The fix is to upgrade TRT-LLM, not to write a new handler.**

In v1.3.0rc13+:
- `NemotronH_Nano_Omni_Reasoning_V3` is registered via `@register_auto_model("NemotronH_Nano_Omni_Reasoning_V3")` (same decorator stack as `NemotronH_Nano_VL_V2`)
- `modeling_parakeet.py` exists and provides `ParakeetExtractor` / `ProjectedParakeet`
- `modeling_radio.py` supports the Conv3D tubelet path for video
- `NemotronHMOE` in `modeling_nemotron_h.py` uses `DeepseekV3Gate` with proper n_group/topk_group/routed_scaling_factor
- The weight mapper `NemotronHHfWeightMapper` handles all V3 MoE weight remapping

**NGC container for v1.3.0rc13:** `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`

### If the user must stay on v1.2.1 (not recommended)

Backporting the V3 handler requires touching the following files:

| File | Change needed |
|------|--------------|
| `tensorrt_llm/_torch/models/modeling_nemotron_h.py` | Already has `DeepseekV3Gate` in v1.2.1 for Super; verify `NemotronHMOE` reads `n_group`/`topk_group` from config. The V1.2.1 version may not read these fields for Nano (it was written for Super first). |
| `tensorrt_llm/_torch/models/modeling_nemotron_nano.py` | Add `@register_auto_model("NemotronH_Nano_Omni_Reasoning_V3")` decorator. Add `sound_context_token_id` handling. |
| `tensorrt_llm/_torch/models/modeling_parakeet.py` | **Must create** — this file does not exist in v1.2.1. Contains `ParakeetExtractor` and `ProjectedParakeet`. |
| `tensorrt_llm/_torch/models/modeling_radio.py` | May need the `num_frames` Conv3D path added if v1.2.1 only has 2D RADIO. |
| `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_h_weight_mapper.py` | Should already handle V3 MoE weights if it reads `moe_intermediate_size` correctly. |

### Reuse map (what you can steal from newer TRT-LLM main branch)

```
modeling_nemotron_nano.py (main)  → primary target; contains full V3 registration + audio
modeling_parakeet.py (main)       → copy verbatim; no V1.2.1 equivalent
modeling_radio.py (main)          → diff against v1.2.1 to find tubelet path additions
modeling_nemotron_h.py (main)     → diff for DeepseekV3Gate wiring to NemotronHMOE
```

Files from NVIDIA-NeMo/Nemotron repo:
```
usage-cookbook/Nemotron-3-Nano-Omni/trtllm_cookbook.ipynb → working serve commands + YAML config
```

### Risk assessment

**Low risk (well understood):**
- MoE routing: DeepseekV3Gate is already in TRT-LLM v1.2.1 (for NemotronH Super). The V3 config params `n_group=8, topk_group=1, routed_scaling_factor=2.5` are standard DeepseekV3Gate inputs.
- MLP projectors: identical 2-layer architecture for both vision and audio. Shape determined by `sound_config.hidden_size` and `llm_config.hidden_size`.
- Weight mapper: the HF→TRT-LLM weight renaming for MoE experts (up_proj→w1, down_proj→w2) is already in the mapper.

**Medium risk (requires testing):**
- Audio chunking: long audio (>30s) is split by `ParakeetExtractor._split_audio_into_clips`. Off-by-one in clip boundary handling could corrupt speech embeddings.
- EVS placeholder merging: `_build_evs_adjusted_context_ids` relies on exact placeholder token ID alignment between the preprocessor and the LLM forward. After EVS, video token counts change dynamically; the merge logic must match exactly.
- NVFP4 MoE quantization: shared experts are FP8, routed experts are NVFP4 E2M1 with FP8 block scales. The `moe_model_config` override in `NemotronHMOE.__init__` must correctly select per-expert quant config from `quant_config_dict`. Mismatch causes silent BF16 allocation → shape error at load.

**High risk (most likely failure point on sm_120):**
- CUTLASS MoE kernel for grouped routing on Blackwell: The Omni TRT-LLM cookbook explicitly says "For NVFP4 on B200, the MoE backend can be changed via `moe_config: backend: TRTLLM`", but for RTX PRO 6000 (sm_120), the correct backend is **CUTLASS**. The Blackwell-specific MoE kernel selection path in TRT-LLM v1.2.1 may not know about the V3 grouped routing signature on sm_120. **Verify `moe_config.backend=CUTLASS` explicitly in your YAML.**
- Conv3D tubelet weights in RADIO: The `video_embedder.weight` tensor has a 5D shape `[out_channels, in_channels, T, H, W]`. If TRT-LLM v1.2.1's RADIO handler only handles 4D patch embeddings, it will silently ignore or fail to load this weight, producing garbage video embeddings.

**Unknowns / not publicly documented:**
- Exact FastConformer hidden size and layer count for the Parakeet encoder embedded in Omni (the standalone parakeet-tdt-0.6b-v2 is 600M parameters but the Omni's `sound_config` may differ slightly from the standalone checkpoint's config).
- Whether `e_score_correction_bias` values are present in the V3 NVFP4 checkpoint (required by `DeepseekV3Gate.load_weights`) — if absent, the gate will error.

---

## 6. Open Questions

1. **Does the RTX PRO 6000 (sm_120) MoE CUTLASS kernel support grouped routing with `topk_group=1` in v1.2.1?** The user's existing NVFP4 MoE experience on this platform (MiniMax-M2 issues) suggests Blackwell CUTLASS MoE path may have edge cases. Test by serving the BF16 checkpoint first before NVFP4.

2. **Is `mamba_ssm_cache_dtype: float32` required for sm_120?** The TRT-LLM cookbook mandates this for correct long-sequence Mamba behavior. With float16 SSM cache, Mamba-2 state accumulation drifts over ~8k tokens.

3. **Audio-from-video is explicitly broken in v1.3.0rc13** (per release notes: "known issues for audio-from-video"). The fix landed in a later RC. If the user needs audio-from-video, check which RC resolved it.

4. **Does the V3 `vision_config` contain RADIO's `video_temporal_patch_size=2` directly, or is it nested inside a `vision_config.args` dict?** The TRT-LLM input processor reads `getattr(vision_config, "video_temporal_patch_size", 1)` — if the V3 HF config stores it differently, it silently falls back to T=1, disabling tubelet compression and doubling video token count.

5. **What is the exact per-layer hybrid pattern for V3?** The 52-layer arrangement (23 Mamba + 23 MoE + 6 Attention) is confirmed but the ordering string is not in any public source. It's in `llm_config.hybrid_override_pattern` in the model's `config.json` on HuggingFace. This is required to correctly route each layer to the right handler in TRT-LLM.

6. **Chunked prefill for video is listed as a known issue** in v1.3.0rc13. The YAML config must have `enable_chunked_prefill: false` if using video inputs. This is not an sm_120-specific issue.

---

## Summary for elsung

**The immediate path forward is upgrading to TRT-LLM v1.3.0rc13 or later**, which has a complete V3 handler. The correct container is `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`.

**Serve command for NVFP4 on sm_120:**
```bash
cat > nano_v3.yaml <<EOF
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.80
  mamba_ssm_cache_dtype: float32
moe_config:
  backend: CUTLASS
max_batch_size: 128
EOF

PYTORCH_ALLOC_CONF=expandable_segments:True \
trtllm-serve serve "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4" \
  --host 0.0.0.0 --port 8000 \
  --trust_remote_code \
  --reasoning_parser nano-v3 \
  --tool_parser qwen3_coder \
  --extra_llm_api_options nano_v3.yaml
```

**Known issues to watch on sm_120:**
- Start with BF16 to validate correctness before NVFP4
- No chunked prefill with video inputs until a later RC
- Audio-from-video requires PyAV and may still have issues in rc13
- `mamba_ssm_cache_dtype: float32` is mandatory for long contexts

If backporting to v1.2.1 is truly required, the minimum set of files to copy from TRT-LLM main is: `modeling_parakeet.py` (new), plus diffs to `modeling_nemotron_nano.py` (V3 registration + audio handling), `modeling_radio.py` (tubelet path), and `modeling_nemotron_h.py` (DeepseekV3Gate wiring for NemotronHMOE on Nano configs specifically).
