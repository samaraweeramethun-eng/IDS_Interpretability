#!/usr/bin/env bash
# ============================================================
#  vm_setup.sh — Bootstrap a fresh GCE VM (Ubuntu 22.04 / Debian 12)
#                with NVIDIA T4 driver + CUDA + full pipeline
#
#  Usage (SSH into your VM, then):
#    curl -sL https://raw.githubusercontent.com/samaraweeramethun-eng/IDS_Interpretability/main/vm_setup.sh | bash
#
#  Or if you already cloned the repo:
#    chmod +x vm_setup.sh && ./vm_setup.sh
#
#  NOTE: If you used a "Deep Learning VM" image from GCE,
#        skip straight to Step 3 — drivers are pre-installed.
# ============================================================
set -euo pipefail

echo "============================================================"
echo "  IDS Interpretability — VM Setup (T4 GPU)"
echo "============================================================"

# ── Step 1: System packages ─────────────────────────────────────────
echo ""
echo ">>> [1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential git curl wget unzip \
    python3 python3-pip python3-venv \
    software-properties-common ca-certificates

# ── Step 2: NVIDIA driver + CUDA (skip if Deep Learning VM) ────────
if ! command -v nvidia-smi &> /dev/null; then
    echo ""
    echo ">>> [2/6] Installing NVIDIA driver + CUDA 12.1..."

    # Add NVIDIA package repo (Ubuntu 22.04 / Debian 12)
    DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')
    ARCH=$(uname -m)

    # Install CUDA keyring
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb" \
        -O /tmp/cuda-keyring.deb 2>/dev/null || \
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" \
        -O /tmp/cuda-keyring.deb

    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq

    # Install CUDA toolkit + driver
    sudo apt-get install -y -qq cuda-toolkit-12-1 cuda-drivers

    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
    source /etc/profile.d/cuda.sh
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

    echo ""
    echo ">>> NVIDIA driver installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "(may need reboot)"
else
    echo ""
    echo ">>> [2/6] NVIDIA driver already present — skipping."
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# ── Step 3: Clone repository ────────────────────────────────────────
echo ""
echo ">>> [3/6] Cloning repository..."
WORK_DIR="$HOME/IDS_Interpretability"
if [ -d "$WORK_DIR" ]; then
    echo "  Repo already exists at $WORK_DIR — pulling latest..."
    cd "$WORK_DIR"
    git pull origin main
else
    git clone https://github.com/samaraweeramethun-eng/IDS_Interpretability.git "$WORK_DIR"
    cd "$WORK_DIR"
fi

# ── Step 4: Python venv + dependencies ──────────────────────────────
echo ""
echo ">>> [4/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo ">>> Installing PyTorch with CUDA 12.1..."
pip install --upgrade pip setuptools wheel -q
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q

echo ">>> Installing project dependencies..."
pip install -r requirements.txt -q
pip install -e . -q

# psutil is needed for RAM monitoring in the pipeline script
pip install psutil -q

# Verify
python -c "
import torch, methun_research
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info(0)
    print(f'✓ VRAM: {total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free')
print(f'✓ methun_research: {methun_research.__file__}')
"

# ── Step 5: Dataset ─────────────────────────────────────────────────
echo ""
echo ">>> [5/6] Checking dataset..."
DATA_DIR="$WORK_DIR/data/cicids2017"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/cicids2017.csv" ]; then
    SIZE=$(du -h "$DATA_DIR/cicids2017.csv" | cut -f1)
    echo "  ✓ Full dataset found: $DATA_DIR/cicids2017.csv ($SIZE)"
else
    echo ""
    echo "  ⚠  Full dataset NOT found at: $DATA_DIR/cicids2017.csv"
    echo ""
    echo "  Upload it using one of these methods:"
    echo ""
    echo "  Option A — SCP from your local machine:"
    echo "    scp cicids2017.csv YOUR_USER@VM_EXTERNAL_IP:$DATA_DIR/"
    echo ""
    echo "  Option B — GCS bucket (recommended for large files):"
    echo "    gsutil cp gs://YOUR_BUCKET/cicids2017.csv $DATA_DIR/"
    echo ""
    echo "  Option C — Google Drive via gdown:"
    echo "    pip install gdown"
    echo "    gdown --id YOUR_GDRIVE_FILE_ID -O $DATA_DIR/cicids2017.csv"
    echo ""
    echo "  The 5,000-row sample is available for smoke testing."
fi

# ── Step 6: Summary ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Activate the environment:"
echo "    cd $WORK_DIR && source .venv/bin/activate"
echo ""
echo "  Run the full pipeline:"
echo "    python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv"
echo ""
echo "  Or with the sample (smoke test):"
echo "    python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017_sample.csv"
echo ""
echo "  Download artifacts when done:"
echo "    scp -r YOUR_USER@VM_IP:~/IDS_Interpretability/artifacts/ ."
echo "============================================================"
