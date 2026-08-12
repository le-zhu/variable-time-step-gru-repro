import math
import os
import re
import csv
import json

import numpy as np

from config import ProcessConfig, settings_to_dict


def build_process_config(
    data_dir,
    dt_norm_mode,
    process_settings,
    input_dt_norm_mode=None,
    update_dt_norm_mode=None,
):
    cfg = settings_to_dict(process_settings)
    resolved_input_dt_norm_mode = input_dt_norm_mode or dt_norm_mode
    resolved_update_dt_norm_mode = update_dt_norm_mode or dt_norm_mode
    cfg.update(
        {
            "dt_norm_mode": dt_norm_mode,
            "input_dt_norm_mode": resolved_input_dt_norm_mode,
            "update_dt_norm_mode": resolved_update_dt_norm_mode,
            "data_dir": data_dir,
            "process_config_path": os.path.join(data_dir, "process_config.json"),
            "train_output_path": os.path.join(data_dir, "train.npy"),
            "val_output_path": os.path.join(data_dir, "val.npy"),
            "origin_val_output_path": os.path.join(data_dir, "origin_val.npy"),
            "origin_test_output_path": os.path.join(data_dir, "origin_test.npy"),
            "train_lengths_path": os.path.join(data_dir, "train_lengths.npy"),
            "val_lengths_path": os.path.join(data_dir, "val_lengths.npy"),
            "origin_val_lengths_path": os.path.join(data_dir, "origin_val_lengths.npy"),
            "origin_test_lengths_path": os.path.join(data_dir, "origin_test_lengths.npy"),
            "train_deltas_path": os.path.join(data_dir, "train_delta_t.npy"),
            "val_deltas_path": os.path.join(data_dir, "val_delta_t.npy"),
            "origin_val_deltas_path": os.path.join(data_dir, "origin_val_delta_t.npy"),
            "origin_test_deltas_path": os.path.join(data_dir, "origin_test_delta_t.npy"),
            "norm_stats_path": os.path.join(data_dir, "norm_stats.json"),
        }
    )
    return ProcessConfig(**cfg)


def find_data_files(cfg):
    b_files, h_files, t_files = {}, {}, {}
    for filename in os.listdir(cfg.input_dir):
        match_b = re.match(cfg.file_pattern_b, filename)
        match_h = re.match(cfg.file_pattern_h, filename)
        match_t = re.match(cfg.file_pattern_t, filename)
        if match_b:
            b_files[match_b.group(1)] = filename
        if match_h:
            h_files[match_h.group(1)] = filename
        if match_t:
            t_files[match_t.group(1)] = filename
    common_ids = sorted(set(b_files.keys()) & set(h_files.keys()) & set(t_files.keys()))
    return [(b_files[idx], h_files[idx], t_files[idx]) for idx in common_ids]


def compute_resample_stride(length, max_len):
    if length <= 0 or max_len <= 1:
        return 1
    return int(max(1, math.ceil(length / max_len)))


def resample_series(series, stride):
    if stride <= 1:
        return series
    return series[::stride]


def compute_db_dt(series, dt):
    if len(series) <= 1:
        return np.zeros((0,), dtype=series.dtype)
    return (series[1:] - series[:-1]) / dt


def split_indices(total_size, val_ratio, test_ratio, seed=None):
    if total_size < 3:
        raise ValueError("dataset size must be >= 3 for train/validation/test split")
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    val_size = max(1, val_size)
    test_size = max(1, test_size)
    train_size = total_size - val_size - test_size
    if train_size < 1:
        train_size = 1
        overflow = val_size + test_size + train_size - total_size
        while overflow > 0 and val_size >= test_size and val_size > 1:
            val_size -= 1
            overflow -= 1
        while overflow > 0 and test_size > 1:
            test_size -= 1
            overflow -= 1
        while overflow > 0 and val_size > 1:
            val_size -= 1
            overflow -= 1
    if val_size < 1:
        raise ValueError("validation set size must be >= 1")
    if test_size < 1:
        raise ValueError("test set size must be >= 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_size)
    test_idx = perm[:test_size]
    val_idx = perm[test_size:test_size + val_size]
    train_idx = perm[test_size + val_size:]
    return train_idx, val_idx, test_idx


def _load_samples(cfg):
    data_file_tuples = find_data_files(cfg)
    if not data_file_tuples:
        raise FileNotFoundError("No paired B/H/T csv files found. Expected names like 'B_*_B/H/T.csv'.")

    samples = []
    print("Reading files and building samples...")
    for b_file, h_file, t_file in data_file_tuples:
        try:
            rows_b = _read_numeric_csv_rows(os.path.join(cfg.input_dir, b_file))
            rows_h = _read_numeric_csv_rows(os.path.join(cfg.input_dir, h_file))
            rows_t = _read_numeric_csv_rows(os.path.join(cfg.input_dir, t_file))
            num_experiments = min(len(rows_b), len(rows_h), len(rows_t))

            for i in range(num_experiments):
                b_series_raw = np.array(rows_b[i], dtype=np.float32)
                h_series_raw = np.array(rows_h[i], dtype=np.float32)
                if len(b_series_raw) != len(h_series_raw) or len(b_series_raw) == 0:
                    continue

                if not rows_t[i]:
                    continue
                t_scalar = np.float32(rows_t[i][0])
                if np.isnan(t_scalar):
                    continue

                samples.append((b_series_raw, h_series_raw, t_scalar))
        except Exception as exc:
            print(f"Error processing {b_file}: {exc}")

    if not samples:
        raise ValueError("No valid samples were processed. Please check the CSV files.")
    return samples


def _read_numeric_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            values = []
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if math.isnan(value):
                    continue
                values.append(value)
            rows.append(values)
    return rows


def _add_resampled(entries, out_list, cfg):
    for b_raw, h_raw, t_scalar in entries:
        stride = compute_resample_stride(len(b_raw), cfg.max_len)
        b_res = resample_series(b_raw, stride)
        h_res = resample_series(h_raw, stride)
        if len(b_res) < cfg.min_len:
            continue
        dt = float(cfg.base_delta_t * stride)
        out_list.append((b_res, h_res, t_scalar, dt))


def _compute_rms_scale(items):
    mean_ms = float(np.mean(items)) if items else 0.0
    return float(np.sqrt(max(mean_ms, 1e-12)))


def _input_mean_square(series):
    input_series = series[:-1] if len(series) > 1 else series
    return float(np.mean(input_series ** 2)) if len(input_series) > 0 else 0.0


def _compute_multiple_dt(dt_norm_mode, resampled_all, std_b):
    if dt_norm_mode == "std_db_dt":
        ms_db_dt_list = []
        for b_res, _h_res, _t_scalar, dt in resampled_all:
            b_norm = b_res / std_b
            db_dt = compute_db_dt(b_norm, dt)
            ms_db_dt_list.append(float(np.mean(db_dt ** 2)) if len(db_dt) > 0 else 0.0)
        return _compute_rms_scale(ms_db_dt_list)

    if dt_norm_mode == "max_dt_to_1":
        max_resampled_dt = max(float(dt) for _b_res, _h_res, _t_scalar, dt in resampled_all)
        if max_resampled_dt <= 0.0:
            raise ValueError(f"Invalid max resampled dt: {max_resampled_dt}")
        return float(1.0 / max_resampled_dt)

    raise ValueError("dt_norm_mode must be 'std_db_dt' or 'max_dt_to_1'")


def _build_xy_item(series_b, series_h, t_scalar, input_dt_scaled, std_b, std_h):
    b_norm = series_b.reshape(-1, 1) / std_b
    h_norm = series_h.reshape(-1, 1) / std_h
    dB_dt = np.zeros_like(b_norm)
    dB_dt_forward = compute_db_dt(b_norm.squeeze(-1), input_dt_scaled).reshape(-1, 1)
    if len(dB_dt_forward) > 0:
        dB_dt[:-1] = dB_dt_forward
    t_feature = np.full_like(b_norm, t_scalar)
    x_item = np.hstack([b_norm, dB_dt, t_feature])
    return np.hstack([x_item, h_norm])


def _pad_xy_items(items, target_len=None):
    if not items:
        return np.zeros((0, 0, 0), dtype=np.float32)
    max_len = max(len(item) for item in items) if target_len is None else target_len
    padded = []
    for item in items:
        p = max_len - len(item)
        if p > 0:
            item = np.pad(item, ((0, p), (0, 0)), "constant", constant_values=0)
        padded.append(item)
    return np.stack(padded)


def _build_dataset(entries, max_len, std_b, std_h, input_multiple_dt, update_multiple_dt):
    xy_items = []
    lengths = []
    deltas = []
    for b_res, h_res, t_scalar, dt in entries:
        input_dt_scaled = float(dt * input_multiple_dt)
        update_dt_scaled = float(dt * update_multiple_dt)
        xy_items.append(_build_xy_item(b_res, h_res, t_scalar, input_dt_scaled, std_b, std_h))
        lengths.append(len(b_res))
        deltas.append(update_dt_scaled)
    return (
        _pad_xy_items(xy_items, max_len),
        np.array(lengths, dtype=np.int64),
        np.array(deltas, dtype=np.float32),
    )


def _build_origin_dataset(raw_entries, std_b, std_h, input_multiple_dt, update_multiple_dt, base_delta_t):
    xy_items = []
    origin_lengths = []
    origin_deltas = []
    for b_raw, h_raw, t_scalar in raw_entries:
        input_dt_scaled = float(base_delta_t * input_multiple_dt)
        update_dt_scaled = float(base_delta_t * update_multiple_dt)
        xy_items.append(_build_xy_item(b_raw, h_raw, t_scalar, input_dt_scaled, std_b, std_h))
        origin_lengths.append(len(b_raw))
        origin_deltas.append(update_dt_scaled)

    return (
        _pad_xy_items(xy_items),
        np.array(origin_lengths, dtype=np.int64),
        np.array(origin_deltas, dtype=np.float32),
    )


def process_data(cfg):
    dt_norm_mode = cfg.dt_norm_mode
    input_dt_norm_mode = cfg.input_dt_norm_mode or dt_norm_mode
    update_dt_norm_mode = cfg.update_dt_norm_mode or dt_norm_mode
    valid_modes = {"std_db_dt", "max_dt_to_1"}
    if dt_norm_mode not in valid_modes:
        raise ValueError("dt_norm_mode must be 'std_db_dt' or 'max_dt_to_1'")
    if input_dt_norm_mode not in valid_modes:
        raise ValueError("input_dt_norm_mode must be 'std_db_dt' or 'max_dt_to_1'")
    if update_dt_norm_mode not in valid_modes:
        raise ValueError("update_dt_norm_mode must be 'std_db_dt' or 'max_dt_to_1'")

    print(
        f"[Preprocess] base_dt={cfg.base_delta_t}, min_len={cfg.min_len}, "
        f"max_len={cfg.max_len}, dt_norm_mode={dt_norm_mode}, "
        f"input_dt_norm_mode={input_dt_norm_mode}, update_dt_norm_mode={update_dt_norm_mode}"
    )
    if os.path.exists(cfg.data_dir):
        raise FileExistsError(
            f"Data folder already exists and will not be overwritten: {cfg.data_dir}"
        )
    os.makedirs(cfg.data_dir)

    samples = _load_samples(cfg)
    train_idx, val_idx, test_idx = split_indices(
        len(samples), cfg.validation_split, cfg.test_split, cfg.split_seed
    )
    train_raw = [samples[i] for i in train_idx]
    val_raw = [samples[i] for i in val_idx]
    test_raw = [samples[i] for i in test_idx]

    resampled_train = []
    resampled_val = []
    _add_resampled(train_raw, resampled_train, cfg)
    _add_resampled(val_raw, resampled_val, cfg)

    if not resampled_train or not resampled_val:
        raise ValueError("No valid samples after resampling. Please check the CSV files.")

    normalization_source = resampled_train
    std_b = _compute_rms_scale([_input_mean_square(b_res) for b_res, _h_res, _t, _dt in normalization_source])
    std_h = _compute_rms_scale([_input_mean_square(h_res) for _b_res, h_res, _t, _dt in normalization_source])
    input_multiple_dt = _compute_multiple_dt(input_dt_norm_mode, normalization_source, std_b)
    update_multiple_dt = _compute_multiple_dt(update_dt_norm_mode, normalization_source, std_b)

    print(
        f"Normalization stats: std_b={std_b:.6g}, std_h={std_h:.6g}, "
        f"input_multiple_dt={input_multiple_dt:.6g}, update_multiple_dt={update_multiple_dt:.6g}"
    )
    print("Normalizing and padding...")

    resampled_max_len = max(len(b_res) for b_res, _h_res, _t_scalar, _dt in (resampled_train + resampled_val))
    train_data, train_lengths, train_deltas = _build_dataset(
        resampled_train, resampled_max_len, std_b, std_h, input_multiple_dt, update_multiple_dt
    )
    val_data, val_lengths, val_deltas = _build_dataset(
        resampled_val, resampled_max_len, std_b, std_h, input_multiple_dt, update_multiple_dt
    )
    origin_val_data, origin_val_lengths, origin_val_deltas = _build_origin_dataset(
        val_raw, std_b, std_h, input_multiple_dt, update_multiple_dt, float(cfg.base_delta_t)
    )
    origin_test_data, origin_test_lengths, origin_test_deltas = _build_origin_dataset(
        test_raw, std_b, std_h, input_multiple_dt, update_multiple_dt, float(cfg.base_delta_t)
    )

    print("Saving files...")
    np.save(cfg.train_output_path, train_data)
    np.save(cfg.val_output_path, val_data)
    np.save(cfg.origin_val_output_path, origin_val_data)
    np.save(cfg.origin_test_output_path, origin_test_data)
    np.save(cfg.train_lengths_path, train_lengths)
    np.save(cfg.val_lengths_path, val_lengths)
    np.save(cfg.origin_val_lengths_path, origin_val_lengths)
    np.save(cfg.origin_test_lengths_path, origin_test_lengths)
    np.save(cfg.train_deltas_path, train_deltas)
    np.save(cfg.val_deltas_path, val_deltas)
    np.save(cfg.origin_val_deltas_path, origin_val_deltas)
    np.save(cfg.origin_test_deltas_path, origin_test_deltas)
    with open(cfg.norm_stats_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "std_b": std_b,
                "std_h": std_h,
                "multiple_dt": update_multiple_dt,
                "input_multiple_dt": input_multiple_dt,
                "update_multiple_dt": update_multiple_dt,
                "dt_norm_mode": dt_norm_mode,
                "input_dt_norm_mode": input_dt_norm_mode,
                "update_dt_norm_mode": update_dt_norm_mode,
                "base_delta_t": cfg.base_delta_t,
            },
            handle,
            indent=2,
        )
    with open(cfg.process_config_path, "w", encoding="utf-8") as handle:
        json.dump(settings_to_dict(cfg), handle, indent=2)
    print(
        f"Done. Train samples: {len(train_data)}, Val samples: {len(val_data)}, "
        f"Test samples: {len(origin_test_data)}, "
        f"padded length: {resampled_max_len}."
    )
