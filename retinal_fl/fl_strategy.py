"""
Custom Flower aggregation strategies:
  - FedBNProx: FedProx + FedBN (skip LayerNorm / BatchNorm aggregation)
  - Integration points for HE and SecAgg
"""

from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import numpy as np

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import config as C


def _is_norm_key(key: str) -> bool:
    """Check if parameter key belongs to LayerNorm / BatchNorm."""
    norm_keywords = ("norm", "bn", "layernorm", "batchnorm", "running_mean", "running_var")
    return any(kw in key.lower() for kw in norm_keywords)


class FedBNProxStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg + FedProx (via config) + FedBN (exclude normalisation layers).
    Optionally integrates HE and/or SecAgg.

    Novelty: In ViT / RETFound, LayerNorm statistics encode domain-specific
    information.  By excluding them from aggregation, each client retains
    its site-specific normalisation — critical for non-IID fundus data
    captured under different imaging protocols.
    """

    def __init__(
        self,
        model_keys: Optional[List[str]] = None,
        he_encryptor=None,
        sec_agg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_keys = model_keys or []
        self.he_encryptor = he_encryptor
        self.sec_agg = sec_agg
        self._round = 0

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round-specific config (FedProx mu, local epochs)."""
        self._round = server_round
        config = {
            "mu": C.FEDPROX_MU,
            "local_epochs": C.LOCAL_EPOCHS,
            "server_round": server_round,
        }
        # Use parent to generate the list of (ClientProxy, FitIns)
        fit_configs = super().configure_fit(server_round, parameters, client_manager)
        # Override config in each FitIns
        updated = []
        for proxy, fit_ins in fit_configs:
            new_ins = fl.common.FitIns(fit_ins.parameters, config)
            updated.append((proxy, new_ins))
        return updated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Collect parameters and sample counts
        all_params = []
        sample_counts = []
        for _, fit_res in results:
            params_np = parameters_to_ndarrays(fit_res.parameters)
            all_params.append(params_np)
            sample_counts.append(fit_res.num_examples)

        total_samples = sum(sample_counts)
        n_layers = len(all_params[0])

        # -------- HE-based aggregation (if enabled) --------
        if self.he_encryptor is not None and C.HE_ENABLED:
            try:
                shapes = [p.shape for p in all_params[0]]
                encrypted_all = []
                for client_params in all_params:
                    enc = self.he_encryptor.encrypt_parameters(client_params)
                    encrypted_all.append(enc)

                weights = [sc / total_samples for sc in sample_counts]
                agg_enc = self.he_encryptor.aggregate_encrypted(encrypted_all, sample_counts)
                aggregated = self.he_encryptor.decrypt_parameters(agg_enc, shapes)

                # Restore norm layers from first client (FedBN)
                if self.model_keys:
                    for i, key in enumerate(self.model_keys):
                        if _is_norm_key(key):
                            aggregated[i] = all_params[0][i]

                agg_params = ndarrays_to_parameters(aggregated)
                metrics = self._aggregate_metrics(results)
                metrics["he_used"] = True
                return agg_params, metrics

            except Exception as e:
                print(f"  [WARN] HE aggregation failed ({e}); falling back to plain.")

        # -------- SecAgg masking (if enabled, without HE) --------
        if self.sec_agg is not None and C.SECAGG_ENABLED:
            try:
                from encryption import SecureAggregator
                shapes = [p.shape for p in all_params[0]]
                masks = self.sec_agg.generate_masks(shapes)

                masked_all = []
                for c, client_params in enumerate(all_params):
                    masked = SecureAggregator.mask_parameters(client_params, masks[c])
                    masked_all.append(masked)

                aggregated = SecureAggregator.aggregate_masked(masked_all, sample_counts)

                if self.model_keys:
                    for i, key in enumerate(self.model_keys):
                        if _is_norm_key(key):
                            aggregated[i] = all_params[0][i]

                agg_params = ndarrays_to_parameters(aggregated)
                metrics = self._aggregate_metrics(results)
                metrics["secagg_used"] = True
                return agg_params, metrics

            except Exception as e:
                print(f"  [WARN] SecAgg failed ({e}); falling back to plain.")

        # -------- Plain weighted average with FedBN --------
        aggregated = []
        for i in range(n_layers):
            skip_norm = self.model_keys and _is_norm_key(self.model_keys[i])
            if skip_norm:
                aggregated.append(all_params[0][i])  # keep client-0's norm
            else:
                weighted_sum = np.zeros_like(all_params[0][i], dtype=np.float64)
                for c in range(len(all_params)):
                    weighted_sum += all_params[c][i].astype(np.float64) * (
                        sample_counts[c] / total_samples
                    )
                aggregated.append(weighted_sum.astype(np.float32))

        agg_params = ndarrays_to_parameters(aggregated)
        metrics = self._aggregate_metrics(results)
        return agg_params, metrics

    @staticmethod
    def _aggregate_metrics(results) -> Dict[str, Scalar]:
        accs = [r.metrics.get("accuracy", 0) for _, r in results if r.metrics]
        losses = [r.metrics.get("loss", 0) for _, r in results if r.metrics]
        return {
            "avg_accuracy": float(np.mean(accs)) if accs else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
        }


# ──────────────── evaluation aggregation fn ──────────────────

def weighted_average(metrics):
    """Aggregate evaluation metrics (accuracy) weighted by sample count."""
    accs = [num * m["accuracy"] for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return {"accuracy": sum(accs) / total if total > 0 else 0.0}
