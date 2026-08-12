import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset


class MagNetDataset(Dataset):
    def __init__(self, data_path, len_path, dt_path, seq_len, seed=None, weight_path=None):
        try:
            data = torch.from_numpy(np.load(data_path)).float()
            if data.ndim != 3 or data.shape[2] < 2:
                raise ValueError("data array must be [N, T, C>=2] with Y in last channel")
            self.X = data[:, :, :-1]
            self.Y_traj = data[:, :, -1:].contiguous()
            self.lengths = np.load(len_path).astype(np.int64)
            self.deltas = np.load(dt_path).astype(np.float32)
            if weight_path and os.path.exists(weight_path):
                self.sample_weights = np.load(weight_path).astype(np.float32)
            else:
                self.sample_weights = None
        except FileNotFoundError as exc:
            raise FileNotFoundError("data files not found. Please run preprocessing first.") from exc
        if len(self.lengths) != len(self.X):
            raise ValueError("lengths file length does not match samples.")
        if len(self.deltas) != len(self.X):
            raise ValueError("dt file length does not match samples.")
        if self.sample_weights is not None and len(self.sample_weights) != len(self.X):
            raise ValueError("sample weights file length does not match samples.")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        real_len = int(self.lengths[idx])
        max_start = real_len - self.seq_len
        start = 0 if max_start <= 0 else int(self.rng.integers(0, max_start + 1))
        end = start + self.seq_len
        if self.sample_weights is None:
            return self.X[idx, start:end], self.Y_traj[idx, start:end], float(self.deltas[idx])
        return self.X[idx, start:end], self.Y_traj[idx, start:end], float(self.deltas[idx]), float(self.sample_weights[idx])


def prepare_training_data(cfg):
    seed = int(cfg.seed)
    train_dataset = MagNetDataset(
        cfg.train_data_path,
        cfg.train_len_path,
        cfg.train_dt_path,
        cfg.seq_len12,
        seed=seed,
        weight_path=getattr(cfg, "train_weight_path", ""),
    )
    val_dataset = MagNetDataset(
        cfg.val_data_path, cfg.val_len_path, cfg.val_dt_path, cfg.seq_len12
    )
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if train_size < 1 or val_size < 1:
        raise ValueError("train/val dataset size must be >= 1")
    val_indices = list(range(val_size))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    return train_dataset, val_dataset, val_indices, train_loader


def prepare_validation_data(cfg):
    return prepare_evaluation_data(
        cfg.val_data_path, cfg.val_len_path, cfg.val_dt_path, cfg.seq_len12, "validation"
    )


def prepare_test_data(cfg):
    return prepare_evaluation_data(
        cfg.test_data_path, cfg.test_len_path, cfg.test_dt_path, cfg.seq_len12, "test"
    )


def prepare_evaluation_data(data_path, len_path, dt_path, seq_len, name):
    val_dataset = MagNetDataset(
        data_path, len_path, dt_path, seq_len
    )
    val_size = len(val_dataset)
    if val_size < 1:
        raise ValueError(f"{name} dataset size must be >= 1")
    return val_dataset, list(range(val_size))
