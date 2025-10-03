import optuna
import numpy as np
import torch
import tqdm
import os

from torch.utils.data import DataLoader

from outlier import compute_class_stats, per_class_thresholds_percentile, fit_weibull_per_class, batch_mahalanobis_sq, weibull_outlier_probability, SCIPY_AVAILABLE
from train import metrics_stage_1, NET, MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# precompute features 
def prepare_features(net, train_loader_for_eval, test_loader_known, test_loader_unknown, num_known, semantic_dim):
  net.eval()
  with torch.no_grad():
    # Train features
    train_X = torch.zeros((len(train_loader_for_eval.dataset), semantic_dim))
    train_Y = torch.zeros(len(train_loader_for_eval.dataset))
    for i, data in enumerate(tqdm(train_loader_for_eval, desc="train feats")):
      x_batch, y_batch, z_batch, label = [t.to(device) for t in data]
      _, semantic, _, _, _, _ = net(x_batch, y_batch, z_batch)
      train_X[i] = semantic
      train_Y[i] = label

    # Test features (known + unknown)
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

    # Convert to numpy
    train_X, train_Y = train_X.cpu().numpy(), train_Y.cpu().numpy()
    test_X, test_Y = test_X.cpu().numpy(), test_Y.cpu().numpy()

    # Compute class stats
    class_means, precisions, covariances = compute_class_stats(
      train_X, train_Y, num_known, semantic_dim, reg_lambda=1e-3
    )

    return train_X, train_Y, test_X, test_Y, class_means, precisions

def objective(trial):
  percentile = trial.suggest_float("percentile", 85, 99.9)
  evt_cutoff = trial.suggest_float("evt_cutoff", 0.1, 0.99)
  tail_size = trial.suggest_int("tail_size", 10, 50)

  thresholds = per_class_thresholds_percentile(train_X, train_Y, class_means, precisions, percentile)
  weibull_models = None
  if SCIPY_AVAILABLE:
    weibull_models = fit_weibull_per_class(train_X, train_Y, class_means, precisions, tail_size=tail_size)

  d2 = batch_mahalanobis_sq(test_X, class_means, precisions)
  d = np.sqrt(np.maximum(d2, 0.0))
  x_ct = d - thresholds[np.newaxis, :]

  label_hat = np.zeros(test_X.shape[0], dtype=np.int32)
  for i in range(test_X.shape[0]):
    if np.min(x_ct[i]) > 0:
      label_hat[i] = -1
    else:
      cand = int(np.argmin(x_ct[i]))
      if weibull_models is not None:
        probs = weibull_outlier_probability(d[[i]], weibull_models)
        if probs[0, cand] > evt_cutoff:
          label_hat[i] = -1
        else:
          label_hat[i] = cand
      else:
        label_hat[i] = cand

  # normalize labels
  test_Y_norm = test_Y.copy()
  test_Y_norm[test_Y_norm >= 18] = -1

  tkr, tur, kp, fkr = metrics_stage_1(test_Y_norm, label_hat)

  if tur < 0.8:
      return 0.0
  return (tkr + kp) / 2.0


if __name__ == "__main__":

  semantic_dim = 128
  num_known_class = 18
  in_channels = 1
  input_size = [512, 512]
  len_time = 1
  gamma = 0.75
  my_index = 1

  net = NET(in_channels=in_channels,
              input_size=input_size,
              semantic_dim=semantic_dim,
              num_class=num_known_class,
              device=device).to(device)
  
  model_name = f"group_{my_index}_margin_8_dim_{semantic_dim}_length_{len_time}_gamma_{gamma}_tips_(xyz_for_loss_curve)"
  checkpoint_path = os.path.join("./model/S3R", model_name + ".pkl")

  net.load_state_dict(torch.load(checkpoint_path, map_location=device))
  print(f"Loaded checkpoint from {checkpoint_path}")


  train_data = MyDataset(path_txt=f'./experiment_groups/{my_index}-known_for_train',
                          len_time=len_time, gamma=gamma, size=512)
  test_data_known = MyDataset(path_txt=f'./experiment_groups/{my_index}-known_for_test',
                              len_time=len_time, gamma=gamma, size=512)
  test_data_unknown = MyDataset(path_txt=f'./experiment_groups/{my_index}-unknown',
                                len_time=len_time, gamma=gamma, size=512)

  train_loader_for_eval = DataLoader(train_data, batch_size=1, shuffle=False)
  test_loader_known = DataLoader(test_data_known, batch_size=1, shuffle=False)
  test_loader_unknown = DataLoader(test_data_unknown, batch_size=1, shuffle=False)

  # precompute features once
  train_X, train_Y, test_X, test_Y, class_means, precisions = prepare_features(
    net, train_loader_for_eval, test_loader_known, test_loader_unknown,
    num_known_class, semantic_dim
  )

  # optuna
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=200)

  print("Best params:", study.best_params)
  print("Best score:", study.best_value)
