# Workspace Audit (T237 – Smart Waste Segregation & Recyclability Classification System)

## 1. Initial Inspection Summary
Before writing application code, the development workspace was audited in accordance with Rule 1 (`INSPECT FIRST`).

### Initial Repository Tree
- `/.env.example`
- `/.gitignore`
- `/index.html`
- `/metadata.json`
- `/package.json`
- `/src/App.tsx` (empty starter template)
- `/src/index.css`
- `/src/main.tsx`
- `/tsconfig.json`
- `/vite.config.ts`

### Git Status
- Not a git repository initially (`git status` returned `NOT_A_GIT_REPO`); initialized git repository for clean tracking.

## 2. System & Runtime Environment
| Property | Measured Value |
| :--- | :--- |
| **OS / Kernel** | Linux `4.19.0` x86_64 (`gVisor` sandboxed container) |
| **CPU** | 2 vCPU cores (`nproc = 2`) |
| **Memory & Root Filesystem** | `4.0 GiB` total RAM shared with memory-backed `/` (`tmpfs` root volume) |
| **GPU Availability** | None (`nvidia-smi` unavailable; CPU-only PyTorch + ONNX Runtime execution) |
| **Node.js / npm** | Node `v22.22.2`, npm `10.9.7` |
| **Python** | System Python `3.11.2`; provisioned isolated `.venv` via `uv` (`Python 3.11.2`) |
| **Network Access** | Outbound HTTPS active (`HTTP 200` to PyPI, GitHub, UCI ML Repository, HuggingFace) |
| **Docker Daemon** | `docker` CLI / daemon not installed inside the gVisor container (`BLOCKED` for live container build/run inside sandbox; full multi-stage `Dockerfile`s and `docker-compose.yml` provided and syntax-audited) |

## 3. Implications for Architecture & ML Training
1. **Shared 4.0 GiB Memory/Tmpfs Ceiling**: Because `/` is mounted on a memory-backed `tmpfs` sharing the 4.0 GiB limit with process RAM, all package installation uses `uv pip install --no-cache` and dataset acquisition (`ml/scripts/download_data.py`) streams and compacts raw images to `256x256` RGB JPEG on extraction while deleting raw `.zip` archives immediately.
2. **CPU-Optimized Transfer Learning & ONNX Runtime**: Training uses `timm` ImageNet-pretrained `efficientnet_b0` (primary) and `mobilenetv3_large_100` (lightweight comparison), exporting the best calibrated model to `artifacts/models/model.onnx` for sub-35ms CPU inference via `onnxruntime`.
3. **Single-Origin Proxy on Port 3000**: In the preview environment, `/server.ts` starts FastAPI (`uvicorn app.main:app --port 8000`) as a managed child process and proxies `/api/*` requests from Port 3000 while serving the Vite React application.
