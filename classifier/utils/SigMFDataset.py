import os
import sigmf
import torch
import numpy as np
from scipy.signal import spectrogram
from sigmf import SigMFFile
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SigMFDataset(Dataset):
  def __init__(self, files, size=512, gamma=1.0, len_time=1):
    self.files = files  # [(path, label)]
    self.size = size    # W - number of frequency bins
    self.len_time = int(size * len_time)  # T - number of time samples (flattened)
    self.gamma = gamma  # fraction of each signal to use

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    path, label = self.files[idx]
    signal, fs = self.load_sigmf_sample(path)

    # crop signal based on gamma
    n = int(len(signal) * self.gamma)
    signal = signal[:n]

    if np.iscomplexobj(signal):
      signal = np.abs(signal)

    # compute spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=self.size)
    Sxx = np.log10(Sxx + 1e-12)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())

    # pad to a fixed shape (no truncation)
    target_time = 512  # choose based on your needs
    T, W = Sxx.shape

    # if T > target_time, we keep it as-is (no truncation)
    if T < target_time:
      pad = np.zeros((target_time, W))
      pad[:T, :] = Sxx
      Sxx = pad

    # convert to tensors
    x = torch.FloatTensor(Sxx)  # [T, W]
    y = x.clone()
    z = x.permute(1, 0)
    x = x.unsqueeze(0)

    return x, y, z, label


  def load_sigmf_sample(self, meta_path):
    smf = sigmf.sigmffile.fromfile(meta_path, skip_checksum=True)
    fs = smf.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    data = smf.read_samples()
    if np.max(np.abs(data)) > 0:
      data /= np.max(np.abs(data))
    return data, fs

def split():
  # TODO change this to the actual path - currently set for the ssh vm
  root_dirs = [
    "/data/2ndPhase/data_dev/Drone_RF_Dataset_Basak",
    "/data/2ndPhase/data_dev/RF_Control_Video_Signal_Recordings_Vuorenmaa"
  ]

  files = []
  class_map = {}
  label_counter = 0

  for root in root_dirs:
    for dirpath, _, filenames in os.walk(root):
      for f in filenames:
        # Match SigMF metadata files
        if f.endswith(".meta") or f.endswith(".sigmf-meta"):
          meta_path = os.path.join(dirpath, f)

          # infer the label from the parent directory name
          parent = os.path.basename(os.path.dirname(meta_path))
          if parent in ["Drone_RF_Dataset_Basak", "RF_Control_Video_Signal_Recordings_Vuorenmaa"]:
            label_name = os.path.splitext(f)[0]  # use filename
          else:
            label_name = parent

          # assign class index
          if label_name not in class_map:
            class_map[label_name] = label_counter
            label_counter += 1

          # store (meta_path, label)
          files.append((meta_path, class_map[label_name]))

  # ensure we found files
  if not files:
    raise RuntimeError("No SigMF .meta or .sigmf-meta files found!")

  # split 70 / 15 / 15
  train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
  val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

  # build datasets
  train_dataset = SigMFDataset(train_files)
  val_dataset = SigMFDataset(val_files)
  test_dataset = SigMFDataset(test_files)

  return train_dataset, val_dataset, test_dataset

# handle pytorch's batching if sizes different with collate_fn
def collate_fn(batch):
  """
  Pads all tensors in the batch to the maximum time (T) and width (W)
  so they can be stacked without truncation.
  """
  xs, ys, zs, labels = zip(*batch)

  # find max dimensions in this batch
  max_T = max(x.shape[-2] for x in xs)  # time dimension
  max_W = max(x.shape[-1] for x in xs)  # frequency dimension

  # pad all tensors to [1, max_T, max_W]
  padded_xs, padded_ys, padded_zs = [], [], []

  for x, y, z in zip(xs, ys, zs):
    _, T, W = x.shape

    pad_T = max_T - T
    pad_W = max_W - W

    # pad (left=0, right=pad_W, top=0, bottom=pad_T)
    x_padded = torch.nn.functional.pad(x, (0, pad_W, 0, pad_T))
    y_padded = torch.nn.functional.pad(y, (0, pad_W, 0, pad_T))
    z_padded = torch.nn.functional.pad(z, (0, pad_T, 0, pad_W))

    padded_xs.append(x_padded)
    padded_ys.append(y_padded)
    padded_zs.append(z_padded)

  # stack into a batch
  x_batch = torch.stack(padded_xs)
  y_batch = torch.stack(padded_ys)
  z_batch = torch.stack(padded_zs)
  label_batch = torch.tensor(labels, dtype=torch.long)

  return x_batch, y_batch, z_batch, label_batch


if __name__ == "__main__":
  from torch.utils.data import DataLoader

  # build datasets
  train_ds, val_ds, test_ds = split()

  # create loaders
  train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
  val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
  test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)


  print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

  # fetch one batch
  x, y, z, label = next(iter(train_loader))

  print("\nShapes:")
  print(f"x: {x.shape}   # Expected [B, 1, T, W] (1 channel for CNN)")
  print(f"y: {y.shape}   # Expected [B, T, W] (time-domain feature)")
  print(f"z: {z.shape}   # Expected [B, W, T] (frequency-domain feature)")
  print(f"label: {label.shape}   # Expected [B] (class indices)")

  # inspect one label and its range
  print(f"\nSample label indices: {label.tolist()}")
  print(f"Unique labels in dataset: {len(set(l for _, l in train_ds.files))}")
