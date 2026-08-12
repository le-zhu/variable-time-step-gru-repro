import json
import os

import torch

from config import WorkflowSettings, make_workflow_settings, settings_to_dict
from models import get_model_entry
from plots import save_test_plots
from process import build_process_config, process_data
import train


RUN_CONFIG_FILENAME = "run_config.json"


def run_workflow(settings):
    if not isinstance(settings, WorkflowSettings):
        raise TypeError("run_workflow expects a WorkflowSettings instance")
    settings = make_workflow_settings(
        model_key=settings.model_key,
        dt_norm_mode=settings.dt_norm_mode,
        run_name=settings.run_name,
        data_name=settings.data_name,
        tasks=settings.tasks,
        process=settings.process,
        train=settings.train,
    )
    model_entry = get_model_entry(settings.model_key)
    run_dir = settings.run_dir
    data_dir = settings.data_dir
    dt_norm_mode = settings.dt_norm_mode
    do_process = settings.tasks.process
    do_pretrain = settings.tasks.pretrain
    do_finetune = settings.tasks.finetune
    do_test = settings.tasks.test
    do_draw = settings.tasks.draw

    if not any([do_process, do_pretrain, do_finetune, do_test, do_draw]):
        raise ValueError("Select at least one task.")

    if do_process and os.path.exists(data_dir):
        raise FileExistsError(
            f"Data folder already exists and will not be overwritten: {data_dir}"
        )

    os.makedirs(run_dir, exist_ok=True)
    run_config_path = save_run_settings(settings)
    print(f"Model directory: {settings.model_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Run config: {run_config_path}")
    print(f"Model: {model_entry.label}")
    print(f"Data processing mode: {dt_norm_mode}")

    if do_process:
        process_cfg = build_process_config(
            data_dir=data_dir,
            dt_norm_mode=dt_norm_mode,
            process_settings=settings.process,
        )
        process_data(process_cfg)

    train_cfg = train.build_train_config(
        run_dir=run_dir,
        data_dir=data_dir,
        model_class=model_entry.model_class,
        train_settings=settings.train,
    )

    model = None
    val_dataset = None
    val_indices = None
    test_dataset = None
    test_indices = None
    history = train.new_train_history()
    did_train = do_pretrain or do_finetune

    if did_train:
        _require_training_data(train_cfg)
        train.init_train_log(train_cfg.train_log_path)
        train.set_reproducibility(train_cfg.seed)
        print(f">>> Start selected training stages [Device: {train_cfg.device}]...")
        train_dataset, val_dataset, val_indices, train_loader = train.prepare_training_data(train_cfg)
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        model = train.create_model(train_cfg)

        if do_finetune and not do_pretrain:
            _load_model_weights(model, train_cfg)

        if do_pretrain:
            model = train.run_pretrain_stage(model, train_loader, train_cfg, history)

        if do_finetune:
            model = train.run_finetune_stage(model, train_loader, val_dataset, val_indices, train_cfg, history)
            _load_model_weights(model, train_cfg)

        train.print_training_summary(history, train_cfg)

    if do_test:
        _require_test_data(train_cfg)
        train.init_train_log(train_cfg.train_log_path)
        if model is None:
            train.set_reproducibility(train_cfg.seed)
            model = train.create_model(train_cfg)
            _load_model_weights(model, train_cfg)
        elif do_finetune and os.path.exists(train_cfg.model_path):
            _load_model_weights(model, train_cfg)
        test_dataset, test_indices = train.prepare_test_data(train_cfg)
        test_error, test_stats = train.evaluate_full_length_loss(
            model, test_dataset, test_indices, train_cfg, desc="  [Test Full]"
        )
        test_p95_error = test_stats.get("p95_row_err", float("nan"))
        train.append_test_final_log(train_cfg.train_log_path, test_error, test_p95_error)
        print(f"Final internal test error: {test_error:.4f}, P95 error: {test_p95_error:.4f}")

    if do_draw:
        _require_test_data(train_cfg)
        if model is None:
            model = train.create_model(train_cfg)
            _load_model_weights(model, train_cfg)
        elif do_finetune and os.path.exists(train_cfg.model_path):
            _load_model_weights(model, train_cfg)

        if test_dataset is None or test_indices is None:
            test_dataset, test_indices = train.prepare_test_data(train_cfg)
        test_result = train.evaluate_on_test(
            model, test_dataset, test_indices, train_cfg, include_predictions=True
        )
        save_test_plots(test_result["batches"], train_cfg)

    print("Workflow complete.")


def save_run_settings(settings):
    os.makedirs(settings.run_dir, exist_ok=True)
    path = os.path.join(settings.run_dir, RUN_CONFIG_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(settings_to_dict(settings), handle, indent=2)
    return path


def _require_training_data(cfg):
    _require_paths(
        [
            cfg.train_data_path,
            cfg.train_len_path,
            cfg.train_dt_path,
            cfg.val_data_path,
            cfg.val_len_path,
            cfg.val_dt_path,
        ],
        "Training data is missing. Run data processing first.",
    )


def _require_validation_data(cfg):
    _require_paths(
        [
            cfg.val_data_path,
            cfg.val_len_path,
            cfg.val_dt_path,
        ],
        "Validation data is missing. Run data processing first.",
    )


def _require_test_data(cfg):
    _require_paths(
        [
            cfg.test_data_path,
            cfg.test_len_path,
            cfg.test_dt_path,
        ],
        "Internal test data is missing. Run data processing first.",
    )


def _require_paths(paths, message):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        missing_text = "\n  ".join(missing)
        raise FileNotFoundError(f"{message}\nMissing files:\n  {missing_text}")


def _load_model_weights(model, cfg):
    model_path = cfg.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        state = torch.load(model_path, map_location=cfg.device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(state)
    print(f"Loaded model weights: {model_path}")
