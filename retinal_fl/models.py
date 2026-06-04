"""
Model architectures for multi-disease retinal classification.

Primary   : RETFound (ViT-Large) + LoRA adapters
Comparison: EfficientNet-B0, ResNet-50, DenseNet-121
"""

import os, copy, math
from collections import OrderedDict

import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download

import config as C

# ──────────────── LoRA layer ─────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation injected into an existing nn.Linear."""

    def __init__(self, original: nn.Linear, rank=C.LORA_RANK, alpha=C.LORA_ALPHA, dropout=C.LORA_DROPOUT):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def inject_lora(model, target_modules=("qkv", "proj", "fc1", "fc2")):
    """Replace matching nn.Linear layers with LoRA wrappers."""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_modules):
                setattr(module, child_name, LoRALinear(child))
    return model


# ──────────────── RETFound + LoRA ────────────────────────────

class RETFoundLoRA(nn.Module):
    """
    RETFound (ViT-Large/16) with LoRA adapters on attention layers.
    Only LoRA params + classifier head are trainable (~2-4 M params).
    """

    def __init__(self, num_classes=C.NUM_CLASSES, pretrained=True):
        super().__init__()

        # 1. Create ViT-Large architecture
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        # 2. Load RETFound weights
        if pretrained:
            self._load_retfound_weights()

        # 3. Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # 4. Inject LoRA into attention projections
        inject_lora(self.backbone)

        # 5. Classification head (trainable)
        hidden = 1024  # ViT-Large hidden dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def _load_retfound_weights(self):
        """Download and load RETFound MAE weights from HuggingFace."""
        cache_dir = os.path.join(C.OUTPUT_DIR, "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, C.RETFOUND_FILENAME)

        if not os.path.exists(local_path):
            print("  Downloading RETFound weights from HuggingFace …")
            try:
                local_path = hf_hub_download(
                    repo_id=C.RETFOUND_HF_REPO,
                    filename=C.RETFOUND_FILENAME,
                    cache_dir=cache_dir,
                )
            except Exception as e:
                print(f"  [WARN] Could not download RETFound weights: {e}")
                print("  Falling back to ImageNet-pretrained ViT-Large.")
                self.backbone = timm.create_model(
                    "vit_large_patch16_224", pretrained=True,
                    num_classes=0, global_pool="avg",
                )
                return

        print(f"  Loading RETFound weights from {local_path}")
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # RETFound checkpoint keys may differ slightly from timm's ViT
        model_state = self.backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            # skip decoder / mask-token / decoder-related keys
            if "decoder" in k or "mask_token" in k:
                continue
            clean_k = k.replace("encoder.", "").replace("module.", "")
            if clean_k in model_state and v.shape == model_state[clean_k].shape:
                filtered[clean_k] = v

        loaded = self.backbone.load_state_dict(filtered, strict=False)
        print(f"  RETFound: loaded {len(filtered)} / {len(model_state)} params "
              f"(missing {len(loaded.missing_keys)}, unexpected {len(loaded.unexpected_keys)})")

    def forward(self, x):
        features = self.backbone(x)       # (B, 1024)
        return self.classifier(features)

    def get_features(self, x):
        return self.backbone(x)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────── comparison models ──────────────────────────

class ComparisonModel(nn.Module):
    """Wrapper for EfficientNet-B0 / ResNet-50 / DenseNet-121."""

    def __init__(self, backbone_name: str, num_classes=C.NUM_CLASSES, pretrained=True):
        super().__init__()
        self.name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ──────────────── factory function ───────────────────────────

def build_model(name="retfound", pretrained=True):
    """
    Build model by name.
    Supported: 'retfound', 'efficientnet_b0', 'resnet50', 'densenet121'
    """
    if name == "retfound":
        model = RETFoundLoRA(pretrained=pretrained)
    else:
        model = ComparisonModel(name, pretrained=pretrained)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model '{name}': {total:,} total params, {trainable:,} trainable "
          f"({trainable/total*100:.1f} %)")
    return model


# ──────────────── state-dict helpers for FL ───────────────────

def get_parameters(model):
    """Return trainable parameters as a list of NumPy arrays."""
    return [v.cpu().detach().numpy() for k, v in model.state_dict().items()
            if any(v.requires_grad for v in [model.state_dict()[k]])]


def set_parameters(model, parameters):
    """Set trainable parameters from a list of NumPy arrays."""
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = OrderedDict()
    param_idx = 0
    for k in keys:
        if state_dict[k].requires_grad or "lora" in k or "classifier" in k:
            new_state[k] = torch.tensor(parameters[param_idx], dtype=state_dict[k].dtype)
            param_idx += 1
        else:
            new_state[k] = state_dict[k]
    model.load_state_dict(new_state, strict=False)


def get_all_parameters_np(model):
    """All parameters (for full FL averaging)."""
    return [v.cpu().detach().numpy() for v in model.state_dict().values()]


def set_all_parameters_np(model, parameters):
    """Set all parameters from numpy arrays."""
    state = model.state_dict()
    keys = list(state.keys())
    for k, p in zip(keys, parameters):
        state[k] = torch.tensor(p)
    model.load_state_dict(state)
