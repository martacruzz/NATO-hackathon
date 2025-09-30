"""
Open-Set Outlier Parameter Tuning
=================================

This script performs a grid search over per-class percentile thresholds and 
EVT (Weibull) cutoff probabilities to optimize open-set detection metrics 
(stage 1). It evaluates how well known and unknown samples are separated 
using Mahalanobis distances and optional EVT tail modeling.

Main Components
---------------
1. **Feature Extraction**:
   - Collects semantic embeddings from a trained network (`net`) for:
     - Training samples (for computing class stats)
     - Test known samples
     - Test unknown samples

2. **Class Statistics**:
   - Computes class means and precision (inverse covariance) matrices.
   - Regularization is applied for numerical stability.

3. **Grid Search**:
   - Iterates over combinations of:
     - Percentiles for distance thresholds (e.g., 90, 95, 99)
     - EVT cutoff probabilities (e.g., 0.1, 0.3, 0.5)
   - For each combination, predicts labels for test samples:
     - Samples exceeding per-class thresholds are labeled unknown (-1)
     - EVT outlier probability is used for further rejection if available

4. **Evaluation**:
   - Metrics computed per configuration:
     - TKR (True Known Rate), TUR (True Unknown Rate)
     - KP (Known Precision), FKR (False Known Rejection)
   - Scores are combined to select the best percentile + cutoff pair.

Inputs & Outputs
----------------
- **Inputs**:
  - `net`: trained model to extract embeddings
  - `train_loader_for_eval`: DataLoader for computing class stats
  - `test_loader_known` / `test_loader_unknown`: DataLoaders for evaluation
  - `num_known`: number of known classes
  - `semantic_dim`: embedding dimension
  - Optional: lists of percentiles and EVT cutoffs

- **Outputs**:
  - Prints metrics for all configurations
  - Returns the best configuration (percentile, EVT cutoff, metrics)

Design Notes
------------
- EVT (Weibull tail fitting) is optional; controlled by `SCIPY_AVAILABLE`.
- Scoring function is flexible; here `(TKR + KP)/2` is used by default.
- Stage 1 metric tuning ensures robust detection of unknown samples 
  before Stage 2 clustering.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from tqdm import tqdm

from outlier import (
    compute_class_stats,
    per_class_thresholds_percentile,
    batch_mahalanobis_sq,
    fit_weibull_per_class,
    weibull_outlier_probability,
    SCIPY_AVAILABLE
)

from train import metrics_stage_1

def tune_outlier_params(
    net, device,
    train_loader_for_eval, test_loader_known, test_loader_unknown,
    num_known, semantic_dim,
    percentiles=[90, 95, 99],
    evt_cutoffs=[0.1, 0.3, 0.5]
):
    """
    Try different combinations of percentile thresholds + EVT cutoffs,
    report metrics for each, and return the best config.
    """

    net.eval()
    with torch.no_grad():
        # ===== 1. collect training semantics =====
        train_X = torch.zeros((len(train_loader_for_eval.dataset), semantic_dim))
        train_Y = torch.zeros(len(train_loader_for_eval.dataset))
        for i, data in enumerate(tqdm(train_loader_for_eval, desc="train feats")):
            x_batch, y_batch, z_batch, label = [t.to(device) for t in data]
            _, semantic, _, _, _, _ = net(x_batch, y_batch, z_batch)
            train_X[i] = semantic
            train_Y[i] = label

        # ===== 2. compute class stats =====
        class_means, precisions, covariances = compute_class_stats(
            train_X, train_Y, num_known, semantic_dim, reg_lambda=1e-3
        )

        # ===== 3. gather test features =====
        test_X = torch.zeros((len(test_loader_known.dataset) + len(test_loader_unknown.dataset), semantic_dim))
        test_Y = torch.zeros(len(test_loader_known.dataset) + len(test_loader_unknown.dataset))
        idx = 0
        for data in tqdm(test_loader_known, desc="test known"):
            x_batch, y_batch, z_batch, label = [t.to(device) for t in data]
            _, semantic, _, _, _, _ = net(x_batch, y_batch, z_batch)
            test_X[idx] = semantic
            test_Y[idx] = label
            idx += 1
        for data in tqdm(test_loader_unknown, desc="test unknown"):
            x_batch, y_batch, z_batch, label = [t.to(device) for t in data]
            _, semantic, _, _, _, _ = net(x_batch, y_batch, z_batch)
            test_X[idx] = semantic
            test_Y[idx] = label
            idx += 1

    # convert to numpy
    train_X, train_Y = train_X.cpu().numpy(), train_Y.cpu().numpy()
    test_X, test_Y = test_X.cpu().numpy(), test_Y.cpu().numpy()

    best_score = -1
    best_cfg = None

    # ===== 4. grid search =====
    for p in percentiles:
        thresholds = per_class_thresholds_percentile(train_X, train_Y, class_means, precisions, p)

        # fit EVT models once per percentile
        weibull_models = None
        if SCIPY_AVAILABLE:
            weibull_models = fit_weibull_per_class(train_X, train_Y, class_means, precisions, tail_size=20)

        # compute distances test->class
        d2 = batch_mahalanobis_sq(test_X, class_means, precisions)
        d = np.sqrt(np.maximum(d2, 0.0))
        x_ct = d - thresholds[np.newaxis, :]

        for cutoff in evt_cutoffs:
            label_hat = np.zeros(test_X.shape[0], dtype=np.int32)
            for i in range(test_X.shape[0]):
                if np.min(x_ct[i]) > 0:
                    label_hat[i] = -1
                else:
                    cand = int(np.argmin(x_ct[i]))
                    if weibull_models is not None:
                        probs = weibull_outlier_probability(d[[i]], weibull_models)
                        if probs[0, cand] > cutoff:
                            label_hat[i] = -1
                        else:
                            label_hat[i] = cand
                    else:
                        label_hat[i] = cand

            # normalize ground truth
            test_Y_norm = test_Y.copy()
            test_Y_norm[test_Y_norm >= num_known] = -1

            tkr, tur, kp, fkr = metrics_stage_1(test_Y_norm, label_hat)
            score = (tkr + kp) / 2.0  # you can pick your own score function

            print(f"percentile={p}, cutoff={cutoff} => TKR={tkr:.3f}, TUR={tur:.3f}, KP={kp:.3f}, FKR={fkr:.3f}, score={score:.3f}")

            if score > best_score:
                best_score = score
                best_cfg = (p, cutoff, (tkr, tur, kp, fkr))

    print("\n==== Best Config ====")
    print(f"percentile={best_cfg[0]}, cutoff={best_cfg[1]}, metrics={best_cfg[2]}")
    return best_cfg
