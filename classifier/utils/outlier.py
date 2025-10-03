"""
Outlier and EVT Utilities for Open-Set RF Signal Classification
===============================================================

This module provides functions for computing class-wise statistics, Mahalanobis 
distances, distance-based thresholds, and Extreme Value Theory (EVT) modeling 
for open-set recognition tasks.

Main Components
---------------

1. **Class Statistics**:
  - compute_class_stats(train_embeddings, train_labels, ...)
    Computes per-class means, covariance matrices (regularized), and precision 
    (inverse covariance) matrices. Supports:
      - Tikhonov regularization via reg_lambda
      - Ledoit-Wolf shrinkage (if sklearn available)
  - Outputs are all numpy arrays (float64) for numerical stability.

2. **Distance Computations**:
   - batch_mahalanobis_sq(X, class_means, precisions)
     Computes squared Mahalanobis distances from each sample to each class mean.
     Vectorized implementation for efficiency.

3. **Per-Class Thresholds**:
   - per_class_thresholds_percentile(...)
     Determines per-class distance thresholds by computing the specified percentile 
     (default 95%) of in-class distances. Returns thresholds in distance domain 
     (not squared).

4. **EVT (Weibull) Modeling**:
   - fit_weibull_per_class(...)
     Fits Weibull distributions to the largest distances (tails) for each class.
     Useful for modeling the probability of extreme distances (outliers).
   - weibull_outlier_probability(dists, weibull_models)
     Converts distances to tail probabilities using fitted Weibull models.
     Outputs probability that a sample is an outlier relative to each class.

Design Choices
--------------
- Mahalanobis distances are computed with class-specific precision matrices 
  to account for anisotropic covariance.
- Regularization ensures numerical stability in covariance inversion.
- EVT is used only in Stage 1 (known vs unknown detection) to refine outlier 
  probabilities; tail fitting uses the largest `tail_size` distances per class.
- Numpy arrays are preferred for all computations to avoid GPU/torch dependencies, 
  except when explicitly passed as torch tensors (converted internally).

Dependencies
------------
- Optional: scipy for Weibull fitting and survival functions.
- Optional: sklearn for Ledoit-Wolf covariance shrinkage.

Usage
-----
Typical workflow:
1. Compute class stats on training embeddings.
2. Compute Mahalanobis distances for test embeddings.
3. Obtain per-class thresholds for Stage 1 rejection.
4. Fit Weibull tails and compute outlier probabilities if EVT is used.
"""


import numpy as np

try:
  from scipy.stats import weibull_min
  SCIPY_AVAILABLE = True
except Exception:
  SCIPY_AVAILABLE = False

try:
  from sklearn.covariance import LedoitWolf
  SKLEARN_LW = True
except Exception:
  SKLEARN_LW = False

def compute_class_stats(train_embeddings, train_labels, num_known, semantic_dim, reg_lambda=1e-3, use_ledoit=False):
  """
  Compute per-class means and precision (inverse covariance) matrices, with regularization.
  Parameters:
    - train_embeddings: (N, D) numpy array or torch tensor
    - train_labels: (N,) array of integer class labels [0..num_known-1]
    - num_known: number of known classes
    - semantic_dim: embedding dimension D
    - reg_lambda: shrinkage factor for covariance regularization (float)
    - use_ledoit: if True and sklearn available, use LedoitWolf shrinkage estimator
  Returns:
    - class_means: (C, D) numpy array
    - precisions: (C, D, D) numpy array of inverse covariances
    - covariances: (C, D, D) numpy array of regularized covariance matrices (useful for diagnostics)
  Note: function returns numpy arrays (float64) for numerical stability.
  """

  # convert to numpy
  if hasattr(train_embeddings, "cpu"):
    X = train_embeddings.cpu().numpy()
  else:
    X = np.array(train_embeddings)

  if hasattr(train_labels, "cpu"):
    Y = train_labels.cpu().numpy()
  else:
    Y = np.array(train_labels)

  C = num_known
  D = semantic_dim
  class_means = np.zeros((C,D), dtype=np.float64)
  precisions = np.zeros((C, D, D), dtype=np.float64)
  covariances = np.zeros((C, D, D), dtype=np.float64)

  for c in range(C):
    cls_index = (Y == c)
    cls_samples = X[cls_index]
    if cls_samples.shape[0] == 0:
      raise ValueError(f"No samples for class {c} to compute statistics.")
    
    # compute mean (class center)
    mu = cls_samples.mean(axis=0).astype(np.float64)
    class_means[c] = mu

    # compute biased covariance matrix (divide by n -> consistent with their original code)
    # rowvar=False means rows are samples
    cov = np.cov(cls_samples, rowvar=False, bias=True).astype(np.float64)

    # regularization / shrinkage:
    if use_ledoit and SKLEARN_LW:
      # ledoit-wolf shrinkage estimator returns the shrunk covariance
      lw = LedoitWolf().fit(cls_samples)
      cov_reg = lw.covariance_.astype(np.float64)
    else:
      cov_reg = (1.0 - reg_lambda) * cov + reg_lambda * np.eye(D, dtype=np.float64)

    covariances[c] = cov_reg

    # compute precision matrix (inverse)
    # pinv for numerical stability
    try:
      prec = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
      # fallback to pseudo-inverse
      prec = np.linalg.pinv(cov_reg)
    precisions[c] = prec

  return class_means, precisions, covariances

def batch_mahalanobis_sq(X, class_means, precisions):
  """
  Compute squared Mahalanobis distances from each sample in X to each class mean.
  Inputs:
    - X: (M, D) numpy array
    - class_means: (C, D)
    - precisions: (C, D, D)
  Output:
    - Dmat: (M, C) array where Dmat[i,c] = d_c^2(x_i)
  This is implemented efficiently using vectorized matrix ops.
  """

  X = np.asarray(X, dtype=np.float64)
  M, D = X.shape
  C = class_means.shape[0]
  Dmat = np.zeros((M, C), dtype=np.float64) # distance matrix

  for c in range(C):
    mu = class_means[c]
    prec = precisions[c]
    V = X - mu # (M, D)
    # compute (V @ prec) elementwise dot with V rows
    tmp = V.dot(prec) # (M, D)
    # row-wise dot product V * tmp, sum accross columns
    Dmat[:, c] = np.einsum('ij,ij->i', V, tmp)

  return Dmat

def per_class_thresholds_percentile(train_embeddings, train_labels, class_means, precisions, percentile=95.0):
  """
  Choose per-class distance thresholds by percentile on in-class distances.
  Returns thresholds in the *distance* domain (not squared), shape: (C,)
  """

  if hasattr(train_embeddings, "cpu"):
    X = train_embeddings.cpu().numpy()
  else:
    X = np.array(train_embeddings)

  if hasattr(train_labels, "cpu"):
    Y = train_labels.cpu().numpy()
  else:
    Y = np.array(train_labels)

  C = class_means.shape[0]
  thresholds = np.zeros(C, dtype=np.float64)

  for c in range(C):
    cls_samples = X[Y == c]
    if cls_samples.shape[0] == 0:
      thresholds[c] = np.inf
      continue

    d2 = batch_mahalanobis_sq(cls_samples, class_means[c:c+1], precisions[c:c+1]).flatten()
    d = np.sqrt(np.maximum(d2, 0.0))
    thresholds[c] = float(np.percentile(d, percentile))
  
  return thresholds


# ------------ EVT (Weibull) utilities ------------
def fit_weibull_per_class(train_embeddings, train_labels, class_means, precisions, tail_size=20):
  """
  Fit Weibull distributions to the top-k distances (the extreme tail) for each class.
  Returns a list of fitted weibull params per class: (c, loc, scale) using scipy.stats.weibull_min
  If scipy is not available, returns None for that class.
  tail_size: how many largest distances to use for fitting (e.g. 20-50)
  """

  if not SCIPY_AVAILABLE:
    print("scipy not available - EVT disabled")
    return [None] * class_means.shape[0]
  
  # gather per-class distances
  if hasattr(train_embeddings, "cpu"):
    X = train_embeddings.cpu().numpy()
  else:
    X = np.array(train_embeddings)

  if hasattr(train_labels, "cpu"):
    Y = train_labels.cpu().numpy()
  else:
    Y = np.array(train_labels)

  C = class_means.shape[0]
  weibull_models = [None] * C

  for c in range(C):
    cls_samples = X[Y == c]
    if cls_samples.shape[0] == 0:
      weibull_models[c] = None
      continue

    d2 = batch_mahalanobis_sq(cls_samples, class_means[c:c+1], precisions[c:c+1]).flatten()
    d = np.sqrt(np.maximum(d2, 0.0))

    # pick largest tail_size distances
    if d.shape[0] < 3:
      weibull_models[c] = None
      continue

    tail_k = min(tail_size, d.shape[0])
    tail_values = np.sort(d)[-tail_k:] # ascending them pick last k

    # fit weibull to tail values
    # force location >= 0 by setting floc=0 to stabilize
    try:
      params = weibull_min.fit(tail_values, floc=0) # (c_shape, loc, scale)
      weibull_models[c] = params
    except Exception:
      # fitting failed
      weibull_models[c] = None

  return weibull_models


def weibull_outlier_probability(dists, weibull_models):
  """
  Convert distances (M,C) or (M,) to outlier probabilities using fitted weibull models.
  dists: if shape (M, C): dists[i, c] = distance of i to class c (not squared)
         if shape (M,): interpreted as distances to a single class
  weibull_models: list length C with params or None
  Returns:
    probs: array shape (M, C) of tail-probabilities p = 1 - CDF_weibull(d)
  """

  if not SCIPY_AVAILABLE:
    raise RuntimeError("scipy required for weibull-based probabilities")
  
  d = np.asarray(dists, dtype=np.float64)

  # if single-class distances given as (M,), expand to (M,1)
  if d.ndim == 1:
    d = d[:, np.newaxis]

  M, C = d.shape
  probs = np.zeros((M,C), dtype=np.float64)

  for c in range(C):
    params = weibull_models[c]
    if params is None:
      # fallback to zero probability (no EVT info)
      probs[:, c] = 0.0
      continue

    c_shape, loc, scale = weibull_models[c]

    # survival function = 1 - cdf
    # weilbull_min.sf returns survival function
    probs[:, c] = weibull_min.sf(d[:, c], c_shape, loc=loc, scale=scale)

  return probs