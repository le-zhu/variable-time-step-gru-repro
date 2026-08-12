from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import os
from typing import Any

from models import get_model_entry, get_model_keys


DEFAULT_MODEL_KEY = "DtScaledGRU"
DEFAULT_DATA_NAME = "N87"
INVALID_FOLDER_NAME_CHARS = set('<>:"/\\|?*')
_MISSING = object()

PROCESSING_MODES = {
    "max_dt_to_1": "Scale max resampled dt to 1",
    "std_db_dt": "Scale dt by RMS dB/dt",
}


@dataclass
class TaskSettings:
    process: bool = True
    pretrain: bool = True
    finetune: bool = True
    test: bool = True
    draw: bool = True


@dataclass
class ProcessSettings:
    file_pattern_b: str = r"N87_(\d+)_B.csv"
    file_pattern_h: str = r"N87_(\d+)_H.csv"
    file_pattern_t: str = r"N87_(\d+)_T.csv"
    # No data ships here; the entry points take --input-dir. This default only
    # names the layout data/README.md describes, one directory per material.
    input_dir: str = os.path.join(os.path.dirname(__file__), "data", "N87")
    base_delta_t: float = 1.0
    min_len: int = 1000
    max_len: int = 4000
    validation_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42


@dataclass
class TrainSettings:
    seed: int = 42
    batch_size: int = 16
    grad_clip_value: float = 20.0
    lr_12: float = 1e-3
    lr_3: float = 1e-3
    scheduler_patience: int = 4
    seq_len12: int = 100
    phase1_teacher_forcing_epochs: int = 50
    warmup_len: int = 10
    phase2_curriculum_transition_epochs: int = 50
    seq_len3: int = 1000
    phase3_finetuning_epochs: int = 150
    phase3_patience: int = 12
    eval_batch_size: int = 2048
    eval_warmup_len: int = 100
    plot_workers: int = 12
    device: str = "auto"
    sdt_penalty_weight: float = 0.0
    sdt_penalty_threshold: float = 0.9


@dataclass
class WorkflowSettings:
    model_key: str = DEFAULT_MODEL_KEY
    dt_norm_mode: str = "max_dt_to_1"
    run_name: str = ""
    data_name: str = DEFAULT_DATA_NAME
    model_dir: str = "runs/DtScaledGRU"
    run_dir: str = "runs/DtScaledGRU"
    data_dir: str = "runs/DtScaledGRU/data/N87"
    tasks: TaskSettings = field(default_factory=TaskSettings)
    process: ProcessSettings = field(default_factory=ProcessSettings)
    train: TrainSettings = field(default_factory=TrainSettings)


@dataclass
class ProcessConfig(ProcessSettings):
    dt_norm_mode: str = ""
    input_dt_norm_mode: str = ""
    update_dt_norm_mode: str = ""
    data_dir: str = ""
    process_config_path: str = ""
    train_output_path: str = ""
    val_output_path: str = ""
    origin_val_output_path: str = ""
    origin_test_output_path: str = ""
    train_lengths_path: str = ""
    val_lengths_path: str = ""
    origin_val_lengths_path: str = ""
    origin_test_lengths_path: str = ""
    train_deltas_path: str = ""
    val_deltas_path: str = ""
    origin_val_deltas_path: str = ""
    origin_test_deltas_path: str = ""
    norm_stats_path: str = ""


@dataclass
class TrainConfig(TrainSettings):
    data_dir: str = ""
    run_dir: str = ""
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    train_len_path: str = ""
    val_len_path: str = ""
    test_len_path: str = ""
    train_dt_path: str = ""
    val_dt_path: str = ""
    test_dt_path: str = ""
    train_weight_path: str = ""
    model_path: str = ""
    train_log_path: str = ""
    model_class: Any = None


PROCESS_CONFIG_KEYS = list(ProcessSettings.__dataclass_fields__.keys())
TRAIN_CONFIG_KEYS = list(TrainSettings.__dataclass_fields__.keys())
DEFAULT_TASKS = asdict(TaskSettings())
DEFAULT_PROCESS_SETTINGS = asdict(ProcessSettings())
DEFAULT_TRAIN_SETTINGS = asdict(TrainSettings())


def get_default_settings(model_key=None):
    valid_models = get_model_keys()
    resolved_model_key = model_key if model_key in valid_models else DEFAULT_MODEL_KEY
    model_entry = get_model_entry(resolved_model_key)
    run_name = make_default_run_name()
    data_name = DEFAULT_DATA_NAME
    model_dir = os.path.normpath(model_entry.default_run_dir)
    return WorkflowSettings(
        model_key=resolved_model_key,
        dt_norm_mode=model_entry.default_dt_norm_mode,
        run_name=run_name,
        data_name=data_name,
        model_dir=model_dir,
        run_dir=build_run_dir(model_dir, run_name),
        data_dir=build_data_dir(model_dir, data_name),
    )


def settings_to_dict(settings):
    if not is_dataclass(settings):
        raise TypeError("settings_to_dict expects a dataclass settings object")
    return asdict(settings)


def settings_from_dict(raw):
    if not isinstance(raw, dict):
        raise TypeError("settings_from_dict expects the current settings JSON object")
    expected = set(WorkflowSettings.__dataclass_fields__)
    unknown = set(raw) - expected
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown WorkflowSettings field(s): {unknown_text}")
    required = {"model_key", "dt_norm_mode", "tasks", "process", "train"}
    missing = required - set(raw)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing WorkflowSettings field(s): {missing_text}")
    return make_workflow_settings(
        model_key=raw["model_key"],
        dt_norm_mode=raw["dt_norm_mode"],
        run_dir=raw.get("run_dir", ""),
        run_name=raw.get("run_name", _MISSING),
        data_name=raw.get("data_name", _MISSING),
        tasks=raw["tasks"],
        process=raw["process"],
        train=raw["train"],
    )


def make_workflow_settings(
    model_key,
    dt_norm_mode,
    run_dir="",
    tasks=None,
    process=None,
    train=None,
    run_name=_MISSING,
    data_name=_MISSING,
):
    if model_key not in get_model_keys():
        valid = ", ".join(sorted(get_model_keys()))
        raise ValueError(f"Unknown model '{model_key}'. Valid models: {valid}")
    if dt_norm_mode not in PROCESSING_MODES:
        valid = ", ".join(sorted(PROCESSING_MODES))
        raise ValueError(f"Unknown data processing mode '{dt_norm_mode}'. Valid modes: {valid}")

    model_entry = get_model_entry(model_key)
    model_dir = os.path.normpath(model_entry.default_run_dir)
    resolved_run_name = (
        make_default_run_name() if run_name is _MISSING else validate_folder_name(run_name, "run_name")
    )
    resolved_data_name = (
        DEFAULT_DATA_NAME if data_name is _MISSING else validate_folder_name(data_name, "data_name")
    )
    return WorkflowSettings(
        model_key=model_key,
        dt_norm_mode=dt_norm_mode,
        run_name=resolved_run_name,
        data_name=resolved_data_name,
        model_dir=model_dir,
        run_dir=build_run_dir(model_dir, resolved_run_name),
        data_dir=build_data_dir(model_dir, resolved_data_name),
        tasks=make_task_settings(DEFAULT_TASKS if tasks is None else tasks),
        process=make_process_settings(DEFAULT_PROCESS_SETTINGS if process is None else process),
        train=make_train_settings(DEFAULT_TRAIN_SETTINGS if train is None else train),
    )


def make_default_run_name():
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def validate_folder_name(value, field_name):
    text = "" if value is None else str(value).strip()
    if text in {"", ".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty folder name.")
    if any(ord(char) < 32 or char in INVALID_FOLDER_NAME_CHARS for char in text):
        invalid = "".join(sorted(INVALID_FOLDER_NAME_CHARS))
        raise ValueError(f"{field_name} must not contain path separators or Windows invalid characters: {invalid}")
    return text


def build_run_dir(model_dir, run_name):
    return os.path.join(model_dir, run_name)


def build_data_dir(model_dir, data_name):
    return os.path.join(model_dir, "data", data_name)


DEFAULT_APP_SETTINGS = get_default_settings(DEFAULT_MODEL_KEY)


def make_task_settings(values):
    return _make_dataclass(TaskSettings, values)


def make_process_settings(values):
    return _make_dataclass(ProcessSettings, values)


def make_train_settings(values):
    return _make_dataclass(TrainSettings, values)


def _make_dataclass(cls, values):
    if is_dataclass(values):
        if not isinstance(values, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(values).__name__}")
        return values
    if not isinstance(values, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}")

    defaults = cls()
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {cls.__name__} field(s): {unknown_text}")

    kwargs = {}
    for key in cls.__dataclass_fields__:
        kwargs[key] = _coerce_like(values.get(key, getattr(defaults, key)), getattr(defaults, key))
    return cls(**kwargs)


def _coerce_like(value, template):
    if value in (None, ""):
        return template
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return template
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return str(value) if isinstance(template, str) else value
