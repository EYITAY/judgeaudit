"""Serve ONE open-weight judge on Modal with vLLM, behind an OpenAI-compatible endpoint.

UNTESTED: written without Modal or GPU access. Check every repo id and flag against the model card first.

Setup (once):
  pip install modal && modal setup
  modal secret create awarejudge-vllm-key VLLM_API_KEY=<make-up-a-long-random-string>
  In the Modal dashboard, set a workspace spending limit BEFORE running anything.

Use (one judge at a time; JUDGE picks the model; it is read when you run modal, so set it on the command line):
  JUDGE=muse modal run scripts/modal_serve.py::download     # fetch weights into a Volume once (billed GPU-free)
  JUDGE=muse modal deploy scripts/modal_serve.py             # prints the endpoint URL, e.g. https://<ws>--awarejudge-muse-serve.modal.run
  python3 scripts/check_endpoint.py --base-url https://<url>/v1 --model <repo id> --api-key-env MODAL_VLLM_KEY --role judge
  ... run the judging (see docs), then STOP the app so nothing keeps billing:
  JUDGE=muse modal app stop awarejudge-muse

Safety: max_containers=1 (never more than one copy), scaledown_window=300 s (GPU stops 5 min after the last request),
timeout 3 h per container. The endpoint requires the key in VLLM_API_KEY.
"""
import os
import subprocess

import modal

JUDGES = {
    # key: (HF repo id, GPU spec, extra vllm args). Confirm each on its model card.
    "muse": ("meta-models/Muse-Glimmer-30B", "H100:1", []),
    "qwen": ("Qwen/Qwen3.8-27B", "H100:1",
             # Qwen thinks by default; turn it off so the reply is the JSON label. Needs a recent vLLM.
             ["--default-chat-template-kwargs", '{"enable_thinking": false}']),
    "nemotron": ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16", "H100:2",
                 ["--tensor-parallel-size", "2", "--trust-remote-code"]),
}
KEY = os.environ.get("JUDGE", "muse")
if KEY not in JUDGES:
    raise SystemExit(f"JUDGE must be one of {list(JUDGES)}")
REPO, GPU, EXTRA = JUDGES[KEY]

# A CUDA *devel* base image, because vLLM's FlashInfer sampler compiles with nvcc at start-up and the plain
# Debian image has none ("Could not find nvcc", Muse start-up log, 31 Oct 2026). Run the `info` function first and,
# if torch's CUDA major version differs from the tag below, set CUDA_TAG to match, e.g. CUDA_TAG=13.0.0-devel-ubuntu22.04
CUDA_TAG = os.environ.get("CUDA_TAG", "12.8.0-devel-ubuntu22.04")
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")   # unpinned: new models need a recent vLLM. Record the version used.
    # FlashInfer's sampler compiles with nvcc at start-up and this image has no CUDA toolkit, so turn it off
    # (vLLM then uses its own sampler). Added after "Could not find nvcc" in the Muse start-up log, 31 Oct 2026.
    # JUDGE and CUDA_TAG are passed in because the container re-reads this file and would otherwise fall back to "muse".
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "JUDGE": KEY, "CUDA_TAG": CUDA_TAG})
)
hf_cache = modal.Volume.from_name("awarejudge-hf-cache", create_if_missing=True)
app = modal.App(f"awarejudge-{KEY}")
secret = modal.Secret.from_name("awarejudge-vllm-key")


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=60 * 60)
def download():
    """CPU only: download the weights once so serving does not pay GPU time for the download."""
    from huggingface_hub import snapshot_download
    snapshot_download(REPO)
    hf_cache.commit()
    print("downloaded", REPO)


@app.function(image=image, timeout=20 * 60)
def info():
    """CPU only, no GPU billed: show the versions that decide whether the image will work."""
    import shutil
    import torch
    import vllm
    print("vllm", vllm.__version__)
    print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
    print("nvcc found at", shutil.which("nvcc"))
    print("CUDA_TAG", CUDA_TAG)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache}, secrets=[secret],
              max_containers=1, scaledown_window=300, timeout=3 * 60 * 60)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    cmd = ["vllm", "serve", REPO, "--served-model-name", REPO, "--host", "0.0.0.0", "--port", "8000",
           "--max-model-len", "16384", "--gpu-memory-utilization", "0.90", *EXTRA]
    # the key comes from the VLLM_API_KEY environment variable in the secret
    subprocess.Popen(cmd)