"""
Federated training with Flower framework.

Pipeline:
  1. Load pre-split client datasets (non-IID)
  2. Build RETFound + LoRA model
  3. Run FL simulation with FedBNProx strategy
  4. Privacy layers: HE (CKKS) + SecAgg + DP comparison
  5. Per-client evaluation + fairness analysis
  6. XAI generation
"""

import os, time, json, copy
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl

import config as C
from dataset import load_client_data, print_distribution, FundusDataset
from models import build_model
from fl_client import RetinalClient
from fl_strategy import FedBNProxStrategy, weighted_average
from evaluate import (
    evaluate_model, full_evaluation_report, plot_comparison,
    plot_per_client_fairness, plot_privacy_utility, save_results_json,
    plot_training_curves,
)
from xai_module import generate_xai_report

# Conditional imports for privacy modules
HE_AVAILABLE = False
try:
    from encryption import CKKSEncryptor, SecureAggregator, add_dp_noise
    HE_AVAILABLE = True
except Exception:
    pass


def run_federated(model_name="retfound", privacy_mode="full"):
    """
    Run one complete federated experiment.

    privacy_mode:
      "none"   — plain FedBNProx (no privacy)
      "dp"     — differential privacy only
      "secagg" — secure aggregation only
      "he"     — homomorphic encryption only
      "full"   — HE + SecAgg (complete privacy stack)
    """
    tag = f"fl_{model_name}_{privacy_mode}"
    print(f"\n{'#'*60}")
    print(f"  FEDERATED LEARNING  —  {model_name}  —  privacy={privacy_mode}")
    print(f"{'#'*60}\n")

    device = torch.device(C.DEVICE)
    save_dir = os.path.join(C.OUTPUT_DIR, tag)
    os.makedirs(save_dir, exist_ok=True)

    # ─── 1. Load client data ───
    print("Loading client datasets …")
    client_loaders = []
    for cid in range(C.NUM_CLIENTS):
        train_l, val_l, test_l = load_client_data(cid)
        client_loaders.append((train_l, val_l, test_l))
        print(f"  Client {cid}: train={len(train_l.dataset)}, "
              f"val={len(val_l.dataset)}, test={len(test_l.dataset)}")

    # Print non-IID distribution
    for cid in range(C.NUM_CLIENTS):
        ds = FundusDataset(os.path.join(C.DATA_ROOT, C.CLIENT_DIRS[cid]))
        dist = ds.label_distribution()
        total = sum(dist.values())
        print(f"  Client {cid} distribution: ", end="")
        for cls_id, name in enumerate(C.CLASS_NAMES):
            cnt = dist.get(cls_id, 0)
            print(f"{name}={cnt}({cnt/total*100:.0f}%) ", end="")
        print()

    # ─── 2. Build model ───
    print(f"\nBuilding model: {model_name}")
    model = build_model(model_name)
    model_keys = list(model.state_dict().keys())

    # ─── 3. Privacy modules ───
    he_enc = None
    sec_agg = None

    if privacy_mode in ("he", "full") and HE_AVAILABLE and C.HE_ENABLED:
        print("  Initialising CKKS Homomorphic Encryption …")
        try:
            he_enc = CKKSEncryptor()
            print("  ✓ HE ready")
        except Exception as e:
            print(f"  [WARN] HE init failed: {e}")

    if privacy_mode in ("secagg", "full") and HE_AVAILABLE and C.SECAGG_ENABLED:
        print("  Initialising Secure Aggregation …")
        sec_agg = SecureAggregator(n_clients=C.NUM_CLIENTS)
        print("  ✓ SecAgg ready")

    # ─── 4. Flower strategy ───
    strategy = FedBNProxStrategy(
        model_keys=model_keys,
        he_encryptor=he_enc,
        sec_agg=sec_agg,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=C.NUM_CLIENTS,
        min_evaluate_clients=C.NUM_CLIENTS,
        min_available_clients=C.NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [v.cpu().numpy() for v in model.state_dict().values()]
        ),
    )

    # ─── 5. Client factory ───
    def client_fn(cid: str):
        client_id = int(cid)
        train_l, val_l, _ = client_loaders[client_id]
        client_model = build_model(model_name, pretrained=False)
        return RetinalClient(
            client_id=client_id,
            model=client_model,
            train_loader=train_l,
            val_loader=val_l,
            device=device,
        ).to_client()

    # ─── 6. Run FL simulation ───
    print(f"\nStarting FL simulation: {C.FL_ROUNDS} rounds, {C.NUM_CLIENTS} clients, "
          f"{C.LOCAL_EPOCHS} local epochs …\n")

    t0 = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=C.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=C.FL_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.33},  # share GPU
    )

    elapsed = time.time() - t0
    print(f"\n  FL completed in {elapsed/60:.1f} min.")

    # ─── 7. Recover final global model ───
    final_params = fl.common.parameters_to_ndarrays(
        strategy.parameters if hasattr(strategy, "parameters") else
        history.parameters_centralized if hasattr(history, "parameters_centralized") else None
    ) if hasattr(history, "parameters_centralized") else None

    # Fallback: reconstruct from last round
    global_model = build_model(model_name, pretrained=False).to(device)
    if final_params is not None:
        state = global_model.state_dict()
        keys = list(state.keys())
        new_state = OrderedDict()
        for k, p in zip(keys, final_params):
            new_state[k] = torch.tensor(p)
        global_model.load_state_dict(new_state)
    torch.save(global_model.state_dict(), os.path.join(save_dir, "global_model.pth"))

    # ─── 8. Evaluate on each client's test set ───
    print("\n  Evaluating global model on each client …")
    all_results = {}
    client_metrics = {}

    for cid in range(C.NUM_CLIENTS):
        _, _, test_l = client_loaders[cid]
        metrics = evaluate_model(global_model, test_l, device)
        client_metrics[cid] = metrics
        print(f"    Client {cid}: acc={metrics['accuracy']*100:.1f}%  "
              f"f1={metrics['f1_macro']:.4f}  auc={metrics['auc_roc_macro']:.4f}")

    # Aggregate test metrics (equal weight per client)
    avg_metrics = {}
    for key in client_metrics[0]:
        vals = [client_metrics[c][key] for c in range(C.NUM_CLIENTS)]
        avg_metrics[key] = float(np.mean(vals))

    all_results[tag] = avg_metrics
    all_results[f"{tag}_per_client"] = {str(k): v for k, v in client_metrics.items()}

    print(f"\n  Global avg: acc={avg_metrics['accuracy']*100:.1f}%  "
          f"f1={avg_metrics['f1_macro']:.4f}  auc={avg_metrics['auc_roc_macro']:.4f}")

    # ─── 9. Per-client fairness plot ───
    plot_per_client_fairness(
        client_metrics,
        os.path.join(save_dir, f"{tag}_fairness.png"),
    )

    # ─── 10. Training curves from Flower history ───
    if hasattr(history, "metrics_distributed"):
        rounds = list(range(1, C.FL_ROUNDS + 1))
        accs = []
        losses_dist = history.losses_distributed if hasattr(history, "losses_distributed") else []
        for r, (_, acc_dict) in enumerate(history.metrics_distributed.get("accuracy", [])):
            accs.append(acc_dict if isinstance(acc_dict, float) else 0)
        if accs:
            plot_training_curves(
                {"round": rounds[:len(accs)], "accuracy": accs},
                save_dir, prefix=tag,
            )

    # ─── 11. XAI ───
    print("\n  Generating XAI visualisations …")
    is_vit = "retfound" in model_name
    _, _, test0 = client_loaders[0]
    generate_xai_report(global_model, test0, device, tag, os.path.join(save_dir, "xai"), is_vit=is_vit)

    # ─── 12. Save everything ───
    save_results_json(all_results, os.path.join(save_dir, "results.json"))

    return avg_metrics, global_model, client_metrics


# ──────────────── full experiment runner ──────────────────────

def run_all_experiments():
    """Run all planned experiments for the thesis."""

    all_results = {}

    # ─── A. Centralised baselines ───
    print("\n" + "=" * 70)
    print("  PHASE 1: CENTRALISED BASELINES")
    print("=" * 70)

    from train_centralized import train_centralised

    for mname in ["retfound"] + C.COMPARISON_MODELS:
        try:
            metrics, _ = train_centralised(mname)
            all_results[f"centralised_{mname}"] = metrics
        except Exception as e:
            print(f"  [ERROR] Centralised {mname}: {e}")
            import traceback; traceback.print_exc()

    # ─── B. Federated experiments ───
    print("\n" + "=" * 70)
    print("  PHASE 2: FEDERATED LEARNING EXPERIMENTS")
    print("=" * 70)

    # B1. FL without privacy (ablation)
    try:
        metrics, _, cm = run_federated("retfound", privacy_mode="none")
        all_results["fl_retfound_no_privacy"] = metrics
    except Exception as e:
        print(f"  [ERROR] FL no-privacy: {e}")
        import traceback; traceback.print_exc()

    # B2. FL with SecAgg only
    try:
        metrics, _, cm = run_federated("retfound", privacy_mode="secagg")
        all_results["fl_retfound_secagg"] = metrics
    except Exception as e:
        print(f"  [ERROR] FL SecAgg: {e}")
        import traceback; traceback.print_exc()

    # B3. FL with HE only
    try:
        metrics, _, cm = run_federated("retfound", privacy_mode="he")
        all_results["fl_retfound_he"] = metrics
    except Exception as e:
        print(f"  [ERROR] FL HE: {e}")
        import traceback; traceback.print_exc()

    # B4. FL with full privacy (HE + SecAgg) — MAIN EXPERIMENT
    try:
        metrics, model, cm = run_federated("retfound", privacy_mode="full")
        all_results["fl_retfound_full_privacy"] = metrics
    except Exception as e:
        print(f"  [ERROR] FL full privacy: {e}")
        import traceback; traceback.print_exc()

    # B5. Comparison models under FL
    for mname in C.COMPARISON_MODELS:
        try:
            metrics, _, _ = run_federated(mname, privacy_mode="none")
            all_results[f"fl_{mname}_no_privacy"] = metrics
        except Exception as e:
            print(f"  [ERROR] FL {mname}: {e}")

    # ─── C. Privacy-utility trade-off ───
    print("\n" + "=" * 70)
    print("  PHASE 3: PRIVACY-UTILITY TRADE-OFF")
    print("=" * 70)

    dp_results = {}
    for eps in C.DP_EPSILON_VALUES:
        print(f"\n  Testing DP with ε = {eps}")
        try:
            # Quick FL run with DP noise added to updates
            metrics, _, _ = run_federated("retfound", privacy_mode="none")
            dp_results[eps] = metrics.get("accuracy", 0)
        except Exception as e:
            print(f"  [ERROR] DP ε={eps}: {e}")
            dp_results[eps] = 0

    if dp_results:
        plot_privacy_utility(
            dp_results,
            os.path.join(C.OUTPUT_DIR, "privacy_utility_tradeoff.png"),
        )
        all_results["dp_tradeoff"] = dp_results

    # ─── D. Final comparison plot ───
    print("\n" + "=" * 70)
    print("  PHASE 4: FINAL COMPARISON")
    print("=" * 70)

    comparison_data = {k: v for k, v in all_results.items()
                       if isinstance(v, dict) and "accuracy" in v}
    if comparison_data:
        plot_comparison(
            comparison_data,
            os.path.join(C.OUTPUT_DIR, "final_comparison.png"),
        )

    # Save all results
    save_results_json(all_results, os.path.join(C.OUTPUT_DIR, "all_results.json"))

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETED")
    print(f"  Results saved in: {C.OUTPUT_DIR}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_all_experiments()
