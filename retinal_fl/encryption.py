"""
Privacy modules:
  1. CKKS Homomorphic Encryption via TenSEAL
  2. Secure Aggregation (mask-based)
  3. Differential Privacy helper (Opacus integration)
"""

import numpy as np
import torch
from typing import List
import config as C

# ─────────────────── TenSEAL CKKS ───────────────────────────

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("[WARN] TenSEAL not installed. HE will be skipped. "
          "Install with: pip install tenseal")


class CKKSEncryptor:
    """
    CKKS homomorphic encryption for model-update vectors.

    Workflow:
      1. Client encrypts its parameter deltas (LoRA weights + classifier).
      2. Server aggregates encrypted vectors WITHOUT decrypting.
      3. Server decrypts the aggregate only.
    """

    def __init__(self):
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("TenSEAL is required for HE.")

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=C.HE_POLY_MOD_DEGREE,
            coeff_mod_bit_sizes=C.HE_COEFF_MOD_BIT_SIZES,
        )
        self.context.global_scale = 2 ** C.HE_SCALE_BITS
        self.context.generate_galois_keys()

        # Serialise a *public-only* context for clients (no secret key)
        self._secret_ctx_bytes = self.context.serialize(save_secret_key=True)
        self._public_ctx_bytes = self.context.serialize(save_secret_key=False)

    # ---- client side ----

    def encrypt_parameters(self, params_np: List[np.ndarray]) -> List[bytes]:
        """Encrypt a list of 1-D numpy arrays (flattened param deltas)."""
        ctx = ts.context_from(self._public_ctx_bytes)
        encrypted = []
        for arr in params_np:
            flat = arr.astype(np.float64).flatten()
            vec = ts.ckks_vector(ctx, flat.tolist())
            encrypted.append(vec.serialize())
        return encrypted

    # ---- server side ----

    def aggregate_encrypted(
        self, all_client_enc: List[List[bytes]], weights: List[float]
    ) -> List[bytes]:
        """
        Aggregate encrypted parameter vectors using weighted average.
        all_client_enc[c][i] = serialised CKKS vector for client c, param i.
        """
        ctx = ts.context_from(self._secret_ctx_bytes)
        n_params = len(all_client_enc[0])
        total_w = sum(weights)
        agg = []

        for i in range(n_params):
            vecs = [ts.lazy_ckks_vector_from(all_client_enc[c][i]) for c in range(len(all_client_enc))]
            for v in vecs:
                v.link_context(ctx)

            # weighted sum
            result = vecs[0] * (weights[0] / total_w)
            for c in range(1, len(vecs)):
                result += vecs[c] * (weights[c] / total_w)

            agg.append(result.serialize())

        return agg

    def decrypt_parameters(self, encrypted_agg: List[bytes], shapes: List[tuple]) -> List[np.ndarray]:
        """Decrypt aggregated vectors and reshape."""
        ctx = ts.context_from(self._secret_ctx_bytes)
        decrypted = []
        for enc_bytes, shape in zip(encrypted_agg, shapes):
            vec = ts.lazy_ckks_vector_from(enc_bytes)
            vec.link_context(ctx)
            flat = np.array(vec.decrypt(), dtype=np.float32)
            size = int(np.prod(shape))
            decrypted.append(flat[:size].reshape(shape))
        return decrypted


# ─────────────── Secure Aggregation (Mask-based) ─────────────

class SecureAggregator:
    """
    Simplified mask-based secure aggregation (Bonawitz et al., CCS 2017).

    Each client generates a random mask.  Masks sum to zero across clients.
    Client sends (update + mask).  Server sums masked updates → masks cancel.
    """

    def __init__(self, n_clients=C.NUM_CLIENTS, seed=C.RANDOM_SEED):
        self.n_clients = n_clients
        self.rng = np.random.RandomState(seed)

    def generate_masks(self, shapes: List[tuple]) -> List[List[np.ndarray]]:
        """
        Returns masks[client][param_idx], where sum over clients = 0 for each param.
        """
        masks = [[None] * len(shapes) for _ in range(self.n_clients)]
        for p_idx, shape in enumerate(shapes):
            client_masks = []
            for c in range(self.n_clients - 1):
                m = self.rng.normal(0, 0.01, size=shape).astype(np.float32)
                client_masks.append(m)
            # last client's mask = -(sum of others)
            last_mask = -sum(client_masks)
            client_masks.append(last_mask.astype(np.float32))
            for c in range(self.n_clients):
                masks[c][p_idx] = client_masks[c]
        return masks

    @staticmethod
    def mask_parameters(params: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Add mask to parameters (client side)."""
        return [p + m for p, m in zip(params, masks)]

    @staticmethod
    def aggregate_masked(all_masked: List[List[np.ndarray]], weights: List[float]) -> List[np.ndarray]:
        """Weighted average of masked parameters — masks cancel out."""
        n_params = len(all_masked[0])
        total_w = sum(weights)
        result = []
        for i in range(n_params):
            weighted_sum = sum(
                all_masked[c][i] * (weights[c] / total_w) for c in range(len(all_masked))
            )
            result.append(weighted_sum)
        return result


# ──────────────── Differential Privacy helper ────────────────

def add_dp_noise(parameters: List[np.ndarray], epsilon: float, delta: float = C.DP_DELTA,
                 sensitivity: float = C.DP_MAX_GRAD_NORM) -> List[np.ndarray]:
    """
    Add calibrated Gaussian noise for (ε,δ)-differential privacy.
    σ = sensitivity × √(2 ln(1.25/δ)) / ε
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noisy = []
    for p in parameters:
        noise = np.random.normal(0, sigma, size=p.shape).astype(np.float32)
        noisy.append(p + noise)
    return noisy


# ──────────────── Privacy-utility trade-off ──────────────────

def evaluate_privacy_utility(model_fn, test_loader, epsilon_values=C.DP_EPSILON_VALUES):
    """
    Sweep ε values and measure accuracy degradation.
    model_fn: callable that returns (model, params_np) for a trained model.
    Returns dict {epsilon: accuracy}.
    """
    results = {}
    model, params = model_fn()
    device = next(model.parameters()).device

    for eps in epsilon_values:
        noisy_params = add_dp_noise(params, eps)
        from models import set_all_parameters_np
        test_model = type(model)().to(device)
        set_all_parameters_np(test_model, noisy_params)
        test_model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = test_model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        results[eps] = correct / total if total > 0 else 0.0
        print(f"  ε = {eps:.1f} → accuracy = {results[eps]*100:.2f}%")

    return results
