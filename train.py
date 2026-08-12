import csv
import os
import random

import numpy as np
import torch
import torch.optim as optim

from config import TrainConfig, settings_to_dict
from data import prepare_test_data, prepare_training_data, prepare_validation_data
from eval import evaluate_full_length_loss, evaluate_on_test, evaluate_on_validation
from rollout import rollout_sequence, sequence_relative_rms
from tqdm import tqdm


TRAIN_LOG_COLUMNS = [
    "phase",
    "epoch",
    "tf_boundary",
    "seq_len",
    "lr",
    "train_loss",
    "val_loss",
    "patience",
    "best_val_loss",
    "final_val_error",
    "final_p95_error",
    "final_test_error",
    "final_test_p95_error",
    "grad_norm_median",
    "grad_norm_p90",
    "clip_ratio",
    "batch_loss_std",
    "prediction_loss",
    "sdt_penalty_loss",
    "sdt_sat_frac_0p9",
    "sdt_u_mean",
    "sdt_u_p95",
]


def build_train_config(run_dir, data_dir, model_class, train_settings):
    values = settings_to_dict(train_settings)
    values["device"] = _resolve_device(values.get("device", "auto"))
    values.update(
        {
            "data_dir": data_dir,
            "run_dir": run_dir,
            "train_data_path": os.path.join(data_dir, "train.npy"),
            "val_data_path": os.path.join(data_dir, "origin_val.npy"),
            "test_data_path": os.path.join(data_dir, "origin_test.npy"),
            "train_len_path": os.path.join(data_dir, "train_lengths.npy"),
            "val_len_path": os.path.join(data_dir, "origin_val_lengths.npy"),
            "test_len_path": os.path.join(data_dir, "origin_test_lengths.npy"),
            "train_dt_path": os.path.join(data_dir, "train_delta_t.npy"),
            "val_dt_path": os.path.join(data_dir, "origin_val_delta_t.npy"),
            "test_dt_path": os.path.join(data_dir, "origin_test_delta_t.npy"),
            "train_weight_path": os.path.join(data_dir, "train_sample_weights.npy"),
            "model_path": os.path.join(run_dir, "model.pth"),
            "train_log_path": os.path.join(run_dir, "train_log.csv"),
            "model_class": model_class,
        }
    )
    return TrainConfig(**values)


def _resolve_device(device):
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def set_reproducibility(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rollout_training_sequence(model, input_seq, h_gt, dt, *, teacher_forcing_boundary, device, collect_sdt=False):
    batch_size, seq_len, _feature_count = input_seq.shape
    h_curr = h_gt[:, 0, :]
    hs = torch.zeros(batch_size, model.hidden_size, device=device)
    preds = [h_curr]
    sdt_values = []

    if collect_sdt and not hasattr(model, "forward_step_with_diagnostics"):
        raise TypeError("sdt penalty requires a model with forward_step_with_diagnostics().")

    step_count = max(seq_len - 1, 0)
    tf_end = min(max(int(teacher_forcing_boundary), 0), step_count)

    for t in range(tf_end):
        if collect_sdt:
            h_curr, hs, diagnostics = model.forward_step_with_diagnostics(
                input_seq[:, t, :], h_gt[:, t, :], hs, dt=dt
            )
            sdt_values.append(diagnostics["sdt"])
        else:
            h_curr, hs = model.forward_step(input_seq[:, t, :], h_gt[:, t, :], hs, dt=dt)
        preds.append(h_curr)
    for t in range(tf_end, step_count):
        if collect_sdt:
            h_curr, hs, diagnostics = model.forward_step_with_diagnostics(
                input_seq[:, t, :], h_curr, hs, dt=dt
            )
            sdt_values.append(diagnostics["sdt"])
        else:
            h_curr, hs = model.forward_step(input_seq[:, t, :], h_curr, hs, dt=dt)
        preds.append(h_curr)

    pred = torch.stack(preds, dim=1)
    target_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    target_mask[:, 0] = False
    diagnostics = {}
    if collect_sdt:
        diagnostics["sdt"] = torch.stack(sdt_values, dim=1) if sdt_values else hs.new_zeros((batch_size, 0, model.hidden_size))
        return pred, target_mask, diagnostics
    return pred, target_mask


def _loader_len(loader):
    try:
        return len(loader)
    except TypeError:
        return None


def _progress_update_interval(loader_len):
    if loader_len is None:
        return 25
    return max(25, loader_len // 20)


def _set_loader_dataset_seq_len(loader, seq_len):
    dataset = loader.dataset
    while not hasattr(dataset, "seq_len") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if not hasattr(dataset, "seq_len"):
        raise TypeError("Training DataLoader dataset does not expose a seq_len attribute.")
    dataset.seq_len = int(seq_len)


def compute_sdt_penalty(diagnostics, target_mask, cfg):
    threshold = float(getattr(cfg, "sdt_penalty_threshold", 0.9))
    empty_stats = {
        "sdt_penalty_loss": 0.0,
        "sdt_sat_frac_0p9": 0.0,
        "sdt_u_mean": 0.0,
        "sdt_u_p95": 0.0,
    }
    sdt = diagnostics.get("sdt") if diagnostics else None
    if sdt is None or sdt.numel() == 0:
        zero = target_mask.float().sum() * 0.0
        return zero, empty_stats

    step_mask = target_mask[:, 1:].unsqueeze(-1).expand_as(sdt)
    if not bool(step_mask.any()):
        zero = sdt.sum() * 0.0
        return zero, empty_stats

    valid_sdt = sdt[step_mask]
    excess = torch.relu(valid_sdt - threshold)
    penalty = (excess * excess).mean()
    with torch.no_grad():
        valid_sdt_float = valid_sdt.detach().float()
        stats = {
            "sdt_penalty_loss": float(penalty.detach().cpu()),
            "sdt_sat_frac_0p9": float((valid_sdt_float > threshold).float().mean().cpu()),
            "sdt_u_mean": float(valid_sdt_float.mean().cpu()),
            "sdt_u_p95": float(torch.quantile(valid_sdt_float, 0.95).cpu()),
        }
    return penalty, stats


def run_epoch(model, loader, optimizer, is_train, tf_boundary, device, cfg, seq_len=None):
    model.train() if is_train else model.eval()
    seq_len_used = cfg.seq_len12 if seq_len is None else seq_len
    loss_tensors = []
    prediction_loss_tensors = []
    sdt_penalty_loss_tensors = []
    sdt_sat_frac_tensors = []
    sdt_u_mean_tensors = []
    sdt_u_p95_tensors = []
    grad_norm_tensors = []
    phase = "Train" if is_train else "Validate"
    loop = tqdm(loader, desc=f"  [{phase}]", leave=False)
    loader_len = _loader_len(loader)
    progress_interval = _progress_update_interval(loader_len)
    sdt_penalty_weight = float(getattr(cfg, "sdt_penalty_weight", 0.0))
    collect_sdt = is_train and sdt_penalty_weight > 0.0
    for batch_index, batch in enumerate(loop):
        if len(batch) == 3:
            input_seq, h_traj_gt, dt_batch = batch
            row_weights = None
        elif len(batch) == 4:
            input_seq, h_traj_gt, dt_batch, row_weights = batch
        else:
            raise ValueError(f"Expected training batch with 3 or 4 items, got {len(batch)}.")
        actual_seq_len = input_seq.shape[1]
        if seq_len_used > actual_seq_len:
            raise ValueError(
                f"Requested seq_len={seq_len_used} exceeds actual batch seq_len={actual_seq_len}. "
                "Check that the phase3 training DataLoader dataset was switched to seq_len3."
            )
        input_seq, h_traj_gt = input_seq.to(device), h_traj_gt.to(device)
        dt = dt_batch.to(device).float().view(-1, 1)
        if row_weights is not None:
            row_weights = row_weights.to(device).float()
        input_seq = input_seq[:, :seq_len_used, :]
        h_traj_gt = h_traj_gt[:, :seq_len_used, :]
        if is_train:
            if collect_sdt:
                pred, target_mask, diagnostics = rollout_training_sequence(
                    model,
                    input_seq,
                    h_traj_gt,
                    dt,
                    teacher_forcing_boundary=tf_boundary,
                    device=device,
                    collect_sdt=True,
                )
            else:
                pred, target_mask = rollout_training_sequence(
                    model,
                    input_seq,
                    h_traj_gt,
                    dt,
                    teacher_forcing_boundary=tf_boundary,
                    device=device,
                )
                diagnostics = {}
        else:
            pred, target_mask = rollout_sequence(
                model,
                input_seq,
                h_traj_gt,
                dt,
                teacher_forcing_boundary=tf_boundary,
                device=device,
            )
            diagnostics = {}
        prediction_loss, _row_err = sequence_relative_rms(pred, h_traj_gt, target_mask, row_weights=row_weights)
        if collect_sdt:
            sdt_penalty_loss, sdt_stats = compute_sdt_penalty(diagnostics, target_mask, cfg)
        else:
            sdt_penalty_loss = prediction_loss * 0.0
            sdt_stats = {
                "sdt_penalty_loss": 0.0,
                "sdt_sat_frac_0p9": 0.0,
                "sdt_u_mean": 0.0,
                "sdt_u_p95": 0.0,
            }
        loss = prediction_loss + sdt_penalty_weight * sdt_penalty_loss
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_value)
            grad_norm_tensors.append(grad_norm.detach())
            optimizer.step()

        loss_tensors.append(loss.detach())
        prediction_loss_tensors.append(prediction_loss.detach())
        sdt_penalty_loss_tensors.append(torch.tensor(sdt_stats["sdt_penalty_loss"]))
        sdt_sat_frac_tensors.append(torch.tensor(sdt_stats["sdt_sat_frac_0p9"]))
        sdt_u_mean_tensors.append(torch.tensor(sdt_stats["sdt_u_mean"]))
        sdt_u_p95_tensors.append(torch.tensor(sdt_stats["sdt_u_p95"]))
        is_last_batch = loader_len is not None and (batch_index + 1) == loader_len
        if batch_index == 0 or is_last_batch or (batch_index + 1) % progress_interval == 0:
            loop.set_postfix(loss=f"{float(loss.detach().cpu()):.3e}", pred=f"{float(prediction_loss.detach().cpu()):.3e}")

    batch_losses = torch.stack(loss_tensors).float().cpu().numpy()
    avg_loss = float(np.mean(batch_losses))
    prediction_losses = torch.stack(prediction_loss_tensors).float().cpu().numpy()
    sdt_penalty_losses = torch.stack(sdt_penalty_loss_tensors).float().cpu().numpy()
    sdt_sat_fracs = torch.stack(sdt_sat_frac_tensors).float().cpu().numpy()
    sdt_u_means = torch.stack(sdt_u_mean_tensors).float().cpu().numpy()
    sdt_u_p95s = torch.stack(sdt_u_p95_tensors).float().cpu().numpy()
    stats = {
        "batch_loss_std": float(np.std(batch_losses)) if batch_losses.size else 0.0,
        "prediction_loss": float(np.mean(prediction_losses)) if prediction_losses.size else avg_loss,
        "sdt_penalty_loss": float(np.mean(sdt_penalty_losses)) if sdt_penalty_losses.size else 0.0,
        "sdt_sat_frac_0p9": float(np.mean(sdt_sat_fracs)) if sdt_sat_fracs.size else 0.0,
        "sdt_u_mean": float(np.mean(sdt_u_means)) if sdt_u_means.size else 0.0,
        "sdt_u_p95": float(np.mean(sdt_u_p95s)) if sdt_u_p95s.size else 0.0,
    }
    if is_train and grad_norm_tensors:
        grad_norms_cpu = torch.stack(grad_norm_tensors).float().cpu()
        finite_grad_norms = grad_norms_cpu[torch.isfinite(grad_norms_cpu)]
        if finite_grad_norms.numel() > 0:
            grad_norms_np = finite_grad_norms.numpy().astype(np.float64)
            clip_count = int((finite_grad_norms > cfg.grad_clip_value).sum().item())
            grad_count = int(finite_grad_norms.numel())
        else:
            grad_norms_np = np.array([], dtype=np.float64)
            clip_count = 0
            grad_count = 0
    else:
        grad_norms_np = np.array([], dtype=np.float64)
        clip_count = 0
        grad_count = 0
    if is_train and grad_norms_np.size:
        stats.update(
            {
                "grad_norm_median": float(np.median(grad_norms_np)),
                "grad_norm_p90": float(np.percentile(grad_norms_np, 90)),
                "clip_ratio": float(clip_count / max(grad_count, 1)),
            }
        )
    return avg_loss, stats


def aggregate_stats(stats_list):
    if not stats_list:
        return {}
    agg = {}
    clip_ratios = [s.get("clip_ratio") for s in stats_list if s and "clip_ratio" in s]
    grad_p90 = [s.get("grad_norm_p90") for s in stats_list if s and "grad_norm_p90" in s]
    loss_std = [s.get("batch_loss_std") for s in stats_list if s and "batch_loss_std" in s]
    if clip_ratios:
        agg["clip_ratio"] = float(sum(clip_ratios) / len(clip_ratios))
    if grad_p90:
        grad_p90_sorted = sorted(grad_p90)
        agg["grad_norm_p90"] = float(grad_p90_sorted[len(grad_p90_sorted) // 2])
    if loss_std:
        agg["batch_loss_std"] = float(sum(loss_std) / len(loss_std))
    return agg


def print_hparam_suggestions(train_loss_hist, val_loss_hist, train_stats, val_stats, cfg):
    suggestions = []
    if train_stats and "clip_ratio" in train_stats:
        clip_ratio = train_stats["clip_ratio"]
        if clip_ratio > 0.5:
            suggestions.append(
                f"Gradient clipping triggered {clip_ratio * 100:.1f}% of steps; consider lowering the learning rate or increasing grad_clip_value."
            )
        elif train_stats["grad_norm_p90"] < 0.2 * cfg.grad_clip_value:
            suggestions.append(
                "Gradient norms are mostly below the clip threshold; if training is slow, consider increasing the learning rate or lowering grad_clip_value."
            )
    if train_loss_hist and val_loss_hist:
        gap = (val_loss_hist[-1] - train_loss_hist[-1]) / max(abs(train_loss_hist[-1]), 1e-8)
        if gap > 0.5:
            suggestions.append("Large gap between train and validation loss; possible overfitting.")
    if len(train_loss_hist) >= 3 and train_loss_hist[-1] > train_loss_hist[-2] > train_loss_hist[-3]:
        suggestions.append(
            "Training loss has increased for 3 epochs; consider lowering the learning rate or checking the data."
        )
    if len(val_loss_hist) >= 3 and len(train_loss_hist) >= 2:
        if val_loss_hist[-1] > val_loss_hist[-2] and train_loss_hist[-1] < train_loss_hist[-2]:
            suggestions.append(
                "Validation loss is rising while training loss is falling; possible overfitting. Consider more regularization/early stopping or reducing model capacity."
            )
    if val_stats and val_loss_hist:
        loss_std = val_stats.get("batch_loss_std", 0.0)
        if loss_std / max(val_loss_hist[-1], 1e-8) > 0.5:
            suggestions.append(
                "Validation loss fluctuates strongly; consider lowering the learning rate or increasing batch_size."
            )
    print("Suggestions:")
    if suggestions:
        for item in suggestions:
            print(f"  - {item}")
    else:
        print("  - No adjustment suggestions (no rules triggered).")


def build_plateau_scheduler(optimizer, cfg):
    user_patience = int(cfg.scheduler_patience)
    if user_patience < 1:
        raise ValueError("scheduler_patience must be >= 1")
    torch_patience = user_patience - 1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=torch_patience,
        factor=0.5,
    )
    return scheduler, torch_patience


def init_train_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        _ensure_train_log_columns(path)
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAIN_LOG_COLUMNS)


def _ensure_train_log_columns(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    header = rows[0]
    missing_columns = [column for column in TRAIN_LOG_COLUMNS if column not in header]
    if not missing_columns:
        return
    header = header + missing_columns
    width = len(header)
    migrated_rows = [header]
    for row in rows[1:]:
        migrated_rows.append(row + [""] * max(width - len(row), 0))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(migrated_rows)


def append_train_log(
    path,
    phase,
    epoch,
    tf_boundary,
    seq_len,
    lr,
    train_loss,
    val_loss,
    patience,
    best_val_loss,
    final_val_error="",
    final_p95_error="",
    final_test_error="",
    final_test_p95_error="",
    train_stats=None,
):
    train_stats = train_stats or {}
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                phase,
                epoch,
                tf_boundary,
                seq_len,
                lr,
                train_loss,
                val_loss,
                patience,
                best_val_loss,
                final_val_error,
                final_p95_error,
                final_test_error,
                final_test_p95_error,
                train_stats.get("grad_norm_median", ""),
                train_stats.get("grad_norm_p90", ""),
                train_stats.get("clip_ratio", ""),
                train_stats.get("batch_loss_std", ""),
                train_stats.get("prediction_loss", ""),
                train_stats.get("sdt_penalty_loss", ""),
                train_stats.get("sdt_sat_frac_0p9", ""),
                train_stats.get("sdt_u_mean", ""),
                train_stats.get("sdt_u_p95", ""),
            ]
        )


def append_test_final_log(path, final_test_error, final_test_p95_error):
    append_train_log(
        path=path,
        phase="test_final",
        epoch="",
        tf_boundary="",
        seq_len="",
        lr="",
        train_loss="",
        val_loss="",
        patience="",
        best_val_loss="",
        final_test_error=final_test_error,
        final_test_p95_error=final_test_p95_error,
    )


def create_model(cfg):
    model_class = cfg.model_class
    return model_class().to(cfg.device)


def new_train_history():
    return {
        "train_loss": [],
        "val_loss": [],
        "train_stats": [],
        "val_stats": [],
    }


def run_pretrain_stage(model, train_loader, cfg, history):
    print("\n" + "=" * 60)
    print(f"PHASE 1: Teacher Forcing (for {cfg.phase1_teacher_forcing_epochs} epochs)")
    print("      - Learning rate is CONSTANT.")
    print("=" * 60)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_12)
    for epoch in range(cfg.phase1_teacher_forcing_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch + 1}/{cfg.phase1_teacher_forcing_epochs} [Phase: 1/3] "
            f"[TF Boundary: {cfg.seq_len12}/{cfg.seq_len12}] [LR: {current_lr:.6g}]"
        )
        train_loss, train_stats = run_epoch(
            model, train_loader, optimizer, True, cfg.seq_len12, cfg.device, cfg
        )
        print(f"  Avg Train Loss: {train_loss:.3e}")
        history["train_loss"].append(train_loss)
        history["train_stats"].append(train_stats)
        append_train_log(
            cfg.train_log_path,
            "phase1",
            epoch + 1,
            cfg.seq_len12,
            cfg.seq_len12,
            current_lr,
            train_loss,
            "",
            "",
            "",
            train_stats=train_stats,
        )
    print("INFO: Phase 1 complete.")

    print("\n" + "=" * 60)
    print(f"PHASE 2: Curriculum Transition (for {cfg.phase2_curriculum_transition_epochs} epochs)")
    print("      - Learning rate is RESET and CONSTANT.")
    print("=" * 60)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_12)
    for epoch_offset in range(cfg.phase2_curriculum_transition_epochs):
        current_epoch_in_phase = epoch_offset + 1
        tf_ratio = 1.0 - (current_epoch_in_phase / cfg.phase2_curriculum_transition_epochs)
        tf_boundary = int(cfg.warmup_len + (cfg.seq_len12 - cfg.warmup_len) * tf_ratio)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {current_epoch_in_phase}/{cfg.phase2_curriculum_transition_epochs} [Phase: 2/3] "
            f"[TF Boundary: {tf_boundary}/{cfg.seq_len12}] [LR: {current_lr:.6g}]"
        )
        train_loss, train_stats = run_epoch(model, train_loader, optimizer, True, tf_boundary, cfg.device, cfg)
        print(f"  Avg Train Loss: {train_loss:.3e}")
        history["train_loss"].append(train_loss)
        history["train_stats"].append(train_stats)
        append_train_log(
            cfg.train_log_path,
            "phase2",
            current_epoch_in_phase,
            tf_boundary,
            cfg.seq_len12,
            current_lr,
            train_loss,
            "",
            "",
            "",
            train_stats=train_stats,
        )
    print("INFO: Phase 2 complete.")
    torch.save(model.state_dict(), cfg.model_path)
    print(f"Pretrain model saved: {cfg.model_path}")
    return model


def run_finetune_stage(model, train_loader, val_dataset, val_indices, cfg, history):
    print("\n" + "=" * 60)
    print(f"PHASE 3: Finetuning (max {cfg.phase3_finetuning_epochs} epochs)")
    print("      - Early stopping is ENABLED.")
    print("=" * 60)
    best_val_loss, best_val_p95, patience_counter = float("inf"), float("nan"), 0
    completed_epochs = 0
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_3)
    scheduler, torch_scheduler_patience = build_plateau_scheduler(optimizer, cfg)
    print(f"Scheduler patience(user)={cfg.scheduler_patience}, torch={torch_scheduler_patience}")
    _set_loader_dataset_seq_len(train_loader, cfg.seq_len3)
    for epoch in range(cfg.phase3_finetuning_epochs):
        completed_epochs = epoch + 1
        tf_boundary = cfg.warmup_len
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch + 1}/{cfg.phase3_finetuning_epochs} [Phase: 3/3] "
            f"[TF Boundary: {tf_boundary}/{cfg.seq_len3}] [LR: {current_lr:.6g}]"
        )
        train_loss, train_stats = run_epoch(
            model, train_loader, optimizer, True, tf_boundary, cfg.device, cfg, seq_len=cfg.seq_len3
        )
        with torch.no_grad():
            val_loss, val_stats = evaluate_full_length_loss(model, val_dataset, val_indices, cfg)
        val_p95 = val_stats.get("p95_row_err", float("nan"))
        print(f"  Avg Train Loss: {train_loss:.3e}, Avg Valid Loss: {val_loss:.3e}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_stats"].append(train_stats)
        history["val_stats"].append(val_stats)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_val_p95, patience_counter = val_loss, val_p95, 0
            torch.save(model.state_dict(), cfg.model_path)
            print(f"  New best model for deployment saved! (Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        append_train_log(
            cfg.train_log_path,
            "phase3",
            epoch + 1,
            tf_boundary,
            cfg.seq_len3,
            current_lr,
            train_loss,
            val_loss,
            patience_counter,
            best_val_loss,
            train_stats=train_stats,
        )
        print(f"  Patience: {patience_counter}/{cfg.phase3_patience}")
        if patience_counter >= cfg.phase3_patience:
            print("INFO: Early stopping triggered in Finetuning phase. Training complete.")
            break
    if completed_epochs == 0:
        with torch.no_grad():
            final_val_error, final_stats = evaluate_full_length_loss(model, val_dataset, val_indices, cfg)
        final_p95_error = final_stats.get("p95_row_err", float("nan"))
    else:
        final_val_error = best_val_loss if np.isfinite(best_val_loss) else history["val_loss"][-1]
        final_p95_error = best_val_p95
        if not np.isfinite(final_p95_error) and history["val_stats"]:
            final_p95_error = history["val_stats"][-1].get("p95_row_err", float("nan"))
    best_val_log = best_val_loss if np.isfinite(best_val_loss) else ""
    append_train_log(
        cfg.train_log_path,
        "finetune_final",
        "",
        "",
        "",
        "",
        "",
        final_val_error,
        "",
        best_val_log,
        final_val_error,
        final_p95_error,
    )
    print(f"Final validation error: {final_val_error:.4f}, P95 error: {final_p95_error:.4f}")
    return model


def print_training_summary(history, cfg):
    train_stats_agg = aggregate_stats(history["train_stats"])
    val_stats_agg = aggregate_stats(history["val_stats"])
    print_hparam_suggestions(
        history["train_loss"],
        history["val_loss"],
        train_stats_agg,
        val_stats_agg,
        cfg,
    )
