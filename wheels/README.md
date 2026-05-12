# Pre-built wheels

## `lmcache-0.4.4-sm120-cp312-cp312-linux_x86_64.whl`

Source: built from [LMCache/LMCache](https://github.com/LMCache/LMCache) at tag `v0.4.4` (commit `6fbec46`), with `TORCH_CUDA_ARCH_LIST="12.0"` to target Blackwell sm_120 exclusively. Build script: [`scripts/build-lmcache-sm120.sh`](../scripts/build-lmcache-sm120.sh).

**Why this exists**: the official `lmcache==0.4.4-cu13` wheel on PyPI ships compiled cubins for **sm_70/75/80/86/89/90/100 only — no sm_120, no PTX**. On Blackwell consumer/workstation GPUs (RTX PRO 6000, RTX 5090, RTX 5080, RTX 5070, RTX 5060) this triggers `cudaErrorNoKernelImageForDevice` from `lmc_ops.multi_layer_kv_transfer` during KV-cache write. This wheel has only sm_120 cubins and works.

## Compatibility matrix

| GPU class | This wheel | Notes |
|---|---|---|
| RTX PRO 6000 Blackwell Workstation (96 GB) | ✅ direct | Verified |
| RTX 5090 / 5080 / 5070 Ti / 5070 / 5060 (sm_120) | ✅ direct | Untested but same arch |
| B200 / GB200 (sm_100, datacenter Blackwell) | ❌ wrong arch | Rebuild via `scripts/build-lmcache-sm120.sh` after changing `TORCH_CUDA_ARCH_LIST` to `"10.0"` |
| H100 / H200 Hopper (sm_90) | ❌ wrong arch | Rebuild with `"9.0"` |
| RTX 4090 / RTX 6000 Ada (sm_89) | ❌ wrong arch | Rebuild with `"8.9"` |
| A100 / RTX A6000 Ampere (sm_80) | ❌ wrong arch | Rebuild with `"8.0"` |

## Python / torch / CUDA versions assumed

- Python 3.12 (`cp312` in wheel name)
- Linux x86_64
- torch 2.11.0 + CUDA 13.x runtime (the wheel uses cu13 cudart at import)
- vLLM ≥ 0.20.1 expected as host integration

## Install

```bash
# Verify environment first
which python3.12
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Expect: 12.0

# Install
pip install wheels/lmcache-0.4.4-sm120-cp312-cp312-linux_x86_64.whl --force-reinstall

# Verify the install has sm_120 cubins
cuobjdump --list-elf $(python3.12 -c 'import lmcache, os; print(os.path.dirname(lmcache.__file__))')/c_ops.cpython-312-x86_64-linux-gnu.so | head -3
# Expect: c_ops.cpython-312-x86_64-linux-gnu.1.sm_120.cubin (or similar)
```

Or use the convenience script: `LLM_HOME=/opt/llm bash scripts/install-prebuilt-lmcache.sh`.

## Verify it actually works

After install, launch a vLLM server with `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'` and send a request. If you see `cudaErrorNoKernelImageForDevice` during decode, the wheel didn't take. If you get a clean response, you're good.

## License

LMCache itself is Apache 2.0 (see https://github.com/LMCache/LMCache). This wheel is a binary build artifact under the same license.
