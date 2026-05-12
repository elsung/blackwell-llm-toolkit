# Vision/Multimodal Engine Strategy — Research Synthesis (2026-05-11)

Three research subagents dug into the latest model families to figure out which engine serves what. Detailed reports in this directory; this is the executive summary.

## Strategic table — model → engine path

| Model family | TRT-LLM path | vLLM path | Notes |
|---|---|---|---|
| **Nemotron-3-Nano-Omni V3** | **v1.3.0rc13+** (native, multimodal handler shipped 2026-04-29) | Works on vLLM ≥0.8 but ~10× slower for multimodal (per user's other rig) | Image + video + audio in one pass. `moe_config.backend=CUTLASS` on sm_120 (not TRTLLM). |
| **Qwen 3.5 / Qwen 3.6 (27B dense, 35A3B MoE)** | **BLOCKED** — needs GDN recurrent-kernel TRT plugin (issue #11674) | **vLLM PR #34110 (v0.17.0+)** — native | Architecture is hybrid GatedDeltaNet (3:1 with full attention), NOT renamed Qwen3-VL. Aliasing would silently ship garbage. |
| **Gemma 4 (31B dense, 26B-A4B MoE)** | **AutoDeploy** (PR #12710, self-contained `modeling_gemma4.py`) — NOT the classic `convert_checkpoint.py` | vLLM PR #38826 (TP=1 only for MoE) | Different from Gemma 3: dual head dims (`global_head_dim=512`, `head_dim=256`), `attention_k_eq_v` (no v_proj on global layers), per-layer embeddings (PLE), softcap=30.0, 128-expert MoE for 26B-A4B. Builtin runtime has transformers-version conflict with NVFP4 (TRT-LLM #12764). |
| **DIY-NVFP4 (any HF model)** | Via modelopt `hf_ptq.py`, output is HF checkpoint w/ `hf_quant_config.json` | Same checkpoint loads via `vllm serve --quantization modelopt` | ~57 min for 27B-class on 96 GB box with `--low_memory_mode`. Peak VRAM ~50 GB. MoE: use `--qformat nvfp4_experts_only`. |

## Architecture-specific gotchas captured

### Nemotron-3-Nano-Omni V3
- **52-layer hybrid**: 23 Mamba-2 SSM + 23 MoE + 6 GQA — interleave in `llm_config.hybrid_override_pattern`
- **MoE = DeepSeek-V3 style**: sigmoid gating + `e_score_correction_bias`, group 128 experts into 8 groups × 16, top-1 group then top-6 within, `routed_scaling_factor=2.5`, plus 1 unconditional shared expert (intermediate=3712). Reuses `DeepseekV3Gate` directly in TRT-LLM rc13's `modeling_nemotron_h.py:NemotronHMOE`.
- **Vision = C-RADIOv4-H**: 653M ViT-H, 16px patches, up to 2048×2048. V3 adds `video_embedder` (Conv3D tubelet fusion across pairs of frames, `video_temporal_patch_size=2`) NOT in V2.
- **Audio = Parakeet-TDT-0.6B-v2 encoder only**: log-mel → 3× stride-2 conv → FastConformer → 2-layer MLP projector. ~12.5 audio tokens/second. Long audio chunked at 30s.
- **sm_120 launch flags**: `moe_config.backend=CUTLASS`, `mamba_ssm_cache_dtype=float32`. Audio-from-video has known bug in rc13 (first request must be video type). No chunked prefill with video inputs.
- **Strategy**: validate BF16 first, then NVFP4.

### Qwen 3.5 / Qwen 3.6
- **Hybrid attention**: 48 GatedDeltaNet layers + 16 full-attention layers (3:1), dispatched via `text_config.layer_types: list[str]`
- **NEW config fields**: `linear_conv_kernel_dim`, `linear_key/value_head_dim`, `linear_num_key/value_heads`, `attn_output_gate=true`, `output_gate_type="swish"`, `mamba_ssm_dtype="float32"`, `mtp_num_hidden_layers=1`, `partial_rotary_factor=0.25`
- **RoPE rename**: `rope_scaling` → `rope_parameters` (causes the `'NoneType' item assignment` error TRT-LLM hit when assuming `rope_scaling` existed)
- **Aliasing to Qwen3-VL is structurally wrong** — GDN layers have different weight shapes and no softmax. Loads but silently outputs garbage.
- **vLLM has native support** (PR #34110, v0.17.0, Feb 2026). Hybrid KV cache manager required.
- **TRT-LLM unblock requires writing a GDN recurrent kernel plugin** (multi-week upstream effort tracked at TRT-LLM #11674).

### Gemma 4
- **Different from Gemma 3**: dual head dim (`global_head_dim=512` for full-attn layers; `head_dim=256` for sliding); `attention_k_eq_v=True` (v_proj absent on global layers, reuses k_proj); `layer_types` explicit dispatch; `layer_scalar` per-layer scalar buffer; Per-Layer Embeddings (PLE) via `vocab_size_per_layer_input`; `final_logit_softcapping=30.0` returns; 26B-A4B has MoE block (128 experts, top-8 + 1 shared)
- **vLLM PR #38826** added handler; TP=1 only for MoE (issue #39595)
- **TRT-LLM via AutoDeploy** (PR #12710), self-contained `modeling_gemma4.py`. Avoid classic `convert_checkpoint.py` (only knows Gemma 2).
- **NVFP4 + builtin runtime bug**: NVFP4 Gemma 4 wants `transformers>=5.5.0`; TRT-LLM builtin ships 4.57.3. AutoDeploy path bypasses this (self-contained).

## ModelOpt NVFP4 DIY workflow

```bash
# In ${LLM_HOME}/venvs/quantize-modelopt (or equivalent)
python hf_ptq.py \
  --pyt_ckpt_path ${LLM_HOME}/models/hf/Qwen__Qwen3.6-27B \
  --qformat nvfp4 \
  --kv_cache_qformat fp8 \
  --export_path ${LLM_HOME}/models/hf/Qwen__Qwen3.6-27B-NVFP4 \
  --calib_size 512 \
  --low_memory_mode
```

For MoE models: use `--qformat nvfp4_experts_only` (what NVIDIA ships for Gemma 4 26B-A4B).

`hf_quant_config.json` produced has:
- `producer.name = "modelopt"`
- `quantization.quant_algo = "NVFP4"`
- `quantization.kv_cache_quant_algo = "FP8"`
- `quantization.group_size = 16`

Output: standard HF checkpoint. Loadable via `vllm serve --quantization modelopt` or TRT-LLM AutoDeploy.

Memory: peak ~75-85 GB without `--low_memory_mode`, under 50 GB with. **Confirmed: 27B-class quantized on RTX PRO 6000 96 GB in ~57 minutes** with 512 calibration samples at 4096 tokens each.

## Path forward (priority-ordered)

1. **Upgrade TRT-LLM to v1.3.0rc13 in a new venv** — unlocks Nemotron Omni V3 (native multimodal, fast). [In progress this session.]
2. **Bench Nemotron Omni V3 on rc13** — image, then video, then audio (after the rc13 first-request-must-be-video workaround). End goal: replace the cloud VLM in Video_Analyzer's pipeline with this local fast inference.
3. **Set up vLLM serving for Qwen 3.5/3.6** — these are vLLM-native; accept that path, don't fight TRT-LLM upstream. Test Qwen 3.6 27B BF16 first, then DIY-NVFP4 of it for faster serving.
4. **Try Gemma 4 31B + 26B-A4B NVFP4 via TRT-LLM AutoDeploy** — NVIDIA-official NVFP4 checkpoints already exist (`nvidia/Gemma-4-31B-IT-NVFP4`, `nvidia/Gemma-4-26B-A4B-NVFP4`).
5. **DIY-NVFP4 toolchain in `runllm`/`makellm`** — wrap modelopt `hf_ptq.py` as a first-class `makellm quantize-nvfp4` command so anyone can quantize without the recipe surgery.

## Anti-patterns to avoid (learnings)

- **DON'T alias new architectures to similar-sounding registered ones** without checking the underlying config + weight shapes. Qwen 3.5/3.6 to Qwen3-VL would have shipped garbage; same risk applies to any future "looks similar" alias.
- **DO check the latest TRT-LLM release (including rc) before writing custom handlers.** v1.3.0rc13 had Omni V3 ready while we were writing a manual port for v1.2.1.
- **DO use research subagents to characterize architecture differences upfront.** Saved multiple weeks of dead-end engineering in this session.
