from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _build_plot_tasks(batches, cfg, out_dir):
    tasks = []
    eval_warmup_len = int(cfg.eval_warmup_len)
    for batch in batches:
        batch_indices = batch["batch_indices"]
        batch_lengths = batch["batch_lengths"]
        batch_err = batch["row_err"]
        pred_np = batch["pred"]
        gt_np = batch["gt"]
        if pred_np is None or gt_np is None:
            continue
        for i, err in enumerate(batch_err):
            if np.isnan(err):
                continue
            real_len = int(batch_lengths[i])
            out_path = os.path.join(out_dir, f"row_{batch_indices[i]}_err_{err:.4f}.png")
            tasks.append(
                {
                    "batch_index": batch_indices[i],
                    "real_len": real_len,
                    "err": float(err),
                    "pred": np.ascontiguousarray(pred_np[i][:real_len]),
                    "gt": np.ascontiguousarray(gt_np[i][:real_len]),
                    "eval_warmup_len": eval_warmup_len,
                    "out_path": out_path,
                }
            )
    return tasks


def _save_plot_task(task):
    real_len = task["real_len"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(task["gt"][:real_len], label="Ground Truth", color="blue")
    ax.plot(task["pred"][:real_len], label="Model Output", linestyle="--", color="orange")
    ax.axvline(x=task["eval_warmup_len"], color="red", linestyle=":", label="Warmup End")
    ax.set_title(f"Row {task['batch_index']} - Err_seq: {task['err']:.4f}")
    ax.set_ylabel("H")
    ax.set_xlabel("Time Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(task["out_path"])
    plt.close(fig)
    return task["out_path"]


def _save_plot_tasks(tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            _save_plot_task(task)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _out_path in executor.map(_save_plot_task, tasks):
            pass


def save_test_plots(batches, cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.run_dir, f"test_results_{timestamp}")
    tasks = _build_plot_tasks(batches, cfg, out_dir)
    if not tasks:
        return None

    os.makedirs(out_dir, exist_ok=True)
    workers = max(1, int(getattr(cfg, "plot_workers", 1)))
    print(f"Test outputs: {out_dir}")
    print(f"Saving {len(tasks)} test plots with {workers} worker(s).")
    _save_plot_tasks(tasks, workers)
    print(f"Test outputs: {out_dir}")
    return out_dir


def save_validation_plots(batches, cfg):
    return save_test_plots(batches, cfg)
