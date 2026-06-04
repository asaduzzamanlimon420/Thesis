"""
Explainable AI (XAI) for fundus disease classification.

1. Grad-CAM++ (for CNN backbones: EfficientNet, ResNet, DenseNet)
2. Attention Rollout (for ViT / RETFound)
3. Privacy-preserving XAI (DP-noised saliency)
4. Per-client vs global model comparison
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config as C

# ──────────────── Grad-CAM++ ─────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ for CNN-based models.
    Works with EfficientNet, ResNet, DenseNet.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    @torch.enable_grad()
    def generate(self, input_tensor, target_class=None):
        """
        Args:
            input_tensor: (1, C, H, W) on model device
            target_class: int or None (auto = predicted class)
        Returns:
            cam: numpy (H, W) in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients[0]       # (C_feat, h, w)
        acts  = self.activations[0]     # (C_feat, h, w)

        # Grad-CAM++ weights
        alpha_num = grads.pow(2)
        alpha_den = 2 * grads.pow(2) + (acts * grads.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha_den = torch.where(alpha_den != 0, alpha_den, torch.ones_like(alpha_den))
        alphas = alpha_num / alpha_den
        weights = (alphas * F.relu(grads)).sum(dim=(1, 2))

        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ──────────────── Attention Rollout (ViT) ────────────────────

class AttentionRollout:
    """
    Attention rollout for Vision Transformers (RETFound).
    Aggregates attention across all transformer blocks.
    """

    def __init__(self, model, head_fusion="mean"):
        """
        model: RETFoundLoRA instance
        head_fusion: 'mean', 'max', or 'min' across attention heads
        """
        self.model = model
        self.head_fusion = head_fusion
        self.attentions = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for blk in self.model.backbone.blocks:
            hook = blk.attn.register_forward_hook(self._save_attention)
            self._hooks.append(hook)

    def _save_attention(self, module, inp, out):
        # timm ViT attention: out is (B, N, C); attn weights stored in module
        # We need to compute attention weights manually
        pass

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def generate(self, input_tensor):
        """
        Returns attention rollout map as (H, W) numpy in [0, 1].
        """
        self.model.eval()
        B, C_in, H, W = input_tensor.shape
        patch_size = 16
        n_patches_h = H // patch_size
        n_patches_w = W // patch_size
        n_patches = n_patches_h * n_patches_w

        # Collect attention maps from all blocks
        attention_maps = []

        def attn_hook(module, inp, out):
            x = inp[0]
            B_a, N, C_a = x.shape
            qkv = module.qkv(x).reshape(B_a, N, 3, module.num_heads, C_a // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = (C_a // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.cpu())

        hooks = []
        for blk in self.model.backbone.blocks:
            hooks.append(blk.attn.register_forward_hook(attn_hook))

        _ = self.model(input_tensor)

        for h in hooks:
            h.remove()

        if not attention_maps:
            return np.ones((H, W), dtype=np.float32)

        # Rollout
        result = torch.eye(attention_maps[0].shape[-1])
        for attn in attention_maps:
            if self.head_fusion == "mean":
                attn_heads = attn.mean(dim=1)[0]
            elif self.head_fusion == "max":
                attn_heads = attn.max(dim=1)[0][0]
            else:
                attn_heads = attn.min(dim=1)[0][0]

            I = torch.eye(attn_heads.shape[-1])
            a = (attn_heads + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result

        # Take CLS token attention to patches
        mask = result[0, 1:]  # skip CLS
        if mask.numel() >= n_patches:
            mask = mask[:n_patches]
        mask = mask.reshape(n_patches_h, n_patches_w).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # Upscale to image size
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((W, H), PILImage.BILINEAR)
        return np.array(mask_img, dtype=np.float32) / 255.0


# ──────────────── Privacy-Preserving XAI ─────────────────────

def privacy_preserving_cam(cam: np.ndarray, epsilon: float = 4.0) -> np.ndarray:
    """
    Add DP noise to saliency map to prevent leaking patient information.
    Lower epsilon = more privacy, more noise.
    """
    sensitivity = 1.0
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / epsilon
    noise = np.random.normal(0, sigma, size=cam.shape).astype(np.float32)
    noisy_cam = cam + noise
    noisy_cam = np.clip(noisy_cam, 0, 1)
    return noisy_cam


# ──────────────── Visualization helpers ──────────────────────

def overlay_cam(image_np, cam, alpha=0.4, colormap=plt.cm.jet):
    """
    Overlay CAM heatmap on original image.
    image_np: (H, W, 3) uint8
    cam: (h, w) float [0, 1] — will be resized
    """
    import cv2
    H, W = image_np.shape[:2]
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H)))
    heatmap = (colormap(cam_resized / 255.0)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def generate_xai_report(model, test_loader, device, model_name, save_dir, is_vit=False):
    """
    Generate XAI visualizations for sample images from each class.
    Saves overlay images and per-class attention analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Get target layer for Grad-CAM++
    if is_vit:
        xai = AttentionRollout(model)
    else:
        # Find last conv layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d,)):
                target_layer = module
        if target_layer is None:
            print("  [WARN] No Conv2d layer found; skipping Grad-CAM.")
            return
        xai = GradCAMPlusPlus(model, target_layer)

    samples_per_class = min(C.XAI_NUM_SAMPLES, 5)
    class_samples = {c: 0 for c in range(C.NUM_CLASSES)}

    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)

    for images, labels in test_loader:
        for i in range(images.size(0)):
            lbl = labels[i].item()
            if class_samples[lbl] >= samples_per_class:
                continue
            class_samples[lbl] += 1

            inp = images[i:i+1].to(device)

            if is_vit:
                cam = xai.generate(inp)
            else:
                cam = xai.generate(inp, target_class=lbl)

            # De-normalise image for display
            img_display = images[i] * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display * 255, 0, 255).astype(np.uint8)

            overlay = overlay_cam(img_display, cam)

            # Also generate privacy-preserving version
            cam_private = privacy_preserving_cam(cam, epsilon=4.0)
            overlay_private = overlay_cam(img_display, cam_private)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_display)
            axes[0].set_title(f"Original ({C.CLASS_NAMES[lbl]})")
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title("Attention Map")
            axes[1].axis("off")
            axes[2].imshow(overlay_private)
            axes[2].set_title("Privacy-Preserving (ε=4)")
            axes[2].axis("off")

            fname = f"{model_name}_{C.CLASS_NAMES[lbl]}_{class_samples[lbl]}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

        if all(v >= samples_per_class for v in class_samples.values()):
            break

    if not is_vit and hasattr(xai, "remove_hooks"):
        xai.remove_hooks()

    print(f"  XAI report saved to {save_dir}")
