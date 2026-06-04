"""
Configuration for Privacy-Preserving Federated Learning
for Multi-Disease Retinal Detection from Fundus Images.

INSTRUCTIONS:
  1. Set DATA_ROOT to the folder that contains client_0/, client_1/, client_2/, combined/
  2. Each client folder must have sub-folders named after classes (see CLASS_NAMES).
  3. Adjust DEVICE / BATCH_SIZE / NUM_WORKERS for your hardware.
"""

import os, torch

# ─────────────────────────── PATHS ───────────────────────────
# >>> CHANGE THIS to your dataset root <<<
# Kaggle example : "/kaggle/input/your-dataset-name"
# Local example  : "C:/Users/you/datasets/retinal"
DATA_ROOT = "./data"

# Each sub-folder inside a client directory must match one of these names exactly.
CLASS_NAMES = ["diabetic_retinopathy", "glaucoma", "macular_scar", "myopia", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

# Client directories (relative to DATA_ROOT)
CLIENT_DIRS = ["client_0", "client_1", "client_2"]
NUM_CLIENTS = len(CLIENT_DIRS)

# Combined directory for centralised baseline
COMBINED_DIR = "combined"

# Where to save all outputs (models, plots, logs)
OUTPUT_DIR = "./outputs"

# ─────────────────────────── HARDWARE ────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# RTX 3060 12 GB → batch 8 with LoRA + mixed-precision is safe.
# Kaggle T4 16 GB  → batch 12 is fine.
# Colab A100 40 GB → batch 32.
BATCH_SIZE = 8
NUM_WORKERS = 2          # Kaggle kernels allow 2; local can be 4
PIN_MEMORY = True
USE_AMP = True           # mixed-precision (float16) — saves ~40 % VRAM

# ─────────────────────────── IMAGE ───────────────────────────
IMG_SIZE = 224            # RETFound input resolution
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (used by RETFound)
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────── MODEL ───────────────────────────
# RETFound  — ViT-Large/16 pretrained on 1.6 M retinal images
# Weights auto-downloaded from HuggingFace on first run.
RETFOUND_HF_REPO = "YukunZhou/RETFound_mae_natureCFP"
RETFOUND_FILENAME = "RETFound_mae_natureCFP.pth"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Comparison backbones (all ImageNet-pretrained via timm)
COMPARISON_MODELS = ["efficientnet_b0", "resnet50", "densenet121"]

# ─────────────────────────── TRAINING ────────────────────────
# --- Centralised baseline ---
CENTRAL_EPOCHS = 30
CENTRAL_LR = 1e-4
WEIGHT_DECAY = 0.01

# --- Federated ---
FL_ROUNDS = 50           # global communication rounds
LOCAL_EPOCHS = 5          # client-side epochs per round
FL_LR = 1e-4
FEDPROX_MU = 0.01         # proximal term coefficient

# ─────────────────────────── NON-IID ─────────────────────────
# Dirichlet concentration — lower = more heterogeneous
DIRICHLET_ALPHA = 0.5
# Additionally apply natural feature-shift by using dataset-origin metadata
USE_FEATURE_SHIFT = True  # extra non-IID dimension (image-quality jitter)

# ─────────────────────────── PRIVACY ─────────────────────────
# Homomorphic Encryption (CKKS via TenSEAL)
HE_ENABLED = True
HE_POLY_MOD_DEGREE = 8192
HE_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
HE_SCALE_BITS = 40

# Differential Privacy (Opacus, for comparison / layering)
DP_ENABLED = True
DP_EPSILON_VALUES = [1.0, 4.0, 6.0, 8.0]  # sweep
DP_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0

# Secure Aggregation
SECAGG_ENABLED = True

# ─────────────────────────── XAI ─────────────────────────────
XAI_NUM_SAMPLES = 50       # images per class for XAI analysis
XAI_METHODS = ["gradcam_pp", "attention_rollout"]

# ─────────────────────────── EVALUATION ──────────────────────
TEST_SPLIT = 0.15          # held-out test fraction
VAL_SPLIT  = 0.10          # validation fraction
RANDOM_SEED = 42
