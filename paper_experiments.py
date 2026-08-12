"""Checked configurations for the experiments reported in the paper."""

COMMON_TRAINING = {
    "seed": 42,
    "batch_size": 16,
    "scheduler_patience": 4,
    "seq_len12": 100,
    "phase1_teacher_forcing_epochs": 50,
    "warmup_len": 10,
    "phase2_curriculum_transition_epochs": 50,
    "seq_len3": 1000,
    "phase3_finetuning_epochs": 150,
    "phase3_patience": 12,
    "eval_batch_size": 2048,
    "eval_warmup_len": 100,
    "plot_workers": 12,
    "sdt_penalty_weight": 0.0,
    "sdt_penalty_threshold": 0.9,
}


N87_EXPERIMENTS = {
    "n87_fixedgru": {
        "material": "N87",
        "paper_name": "FixedGRU",
        "model_key": "StandardGRU",
        "dt_norm_mode": "max_dt_to_1",
        "max_len": 40000,
        "lr_12": 0.001,
        "lr_3": 0.002,
        "grad_clip_value": 20.0,
    },
    "n87_dtscaledgru": {
        "material": "N87",
        "paper_name": "DtScaledGRU",
        "model_key": "DtScaledGRU",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.006,
        "grad_clip_value": 20.0,
    },
    "n87_rategru": {
        "material": "N87",
        "paper_name": "RateGRU",
        "model_key": "SoftplusRateExpCapGRU",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.006,
        "grad_clip_value": 20.0,
    },
    "n87_hidden32": {
        "material": "N87",
        "paper_name": "Hidden32",
        "model_key": "SoftplusRateExpCapGRUHidden32",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.004,
        "grad_clip_value": 20.0,
    },
    "n87_no_h": {
        "material": "N87",
        "paper_name": "No-H",
        "model_key": "SoftplusRateExpCapGRUNoH",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.008,
        "grad_clip_value": 20.0,
    },
    # The joint ablation of time-step normalization and limiting functions.
    # Both normalization schemes are reported for each limiter, so both are
    # released; the expcap rows of that table are n87_dtscaledgru above and
    # n87_rategru / n87_rategru_max_step.
    "n87_dtscaledgru_max_step": {
        "material": "N87",
        "paper_name": "DtScaledGRU",
        "model_key": "DtScaledGRU",
        "dt_norm_mode": "max_dt_to_1",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.002,
        "grad_clip_value": 20.0,
    },
    "n87_rategru_max_step": {
        "material": "N87",
        "paper_name": "RateGRU",
        "model_key": "SoftplusRateExpCapGRU",
        "dt_norm_mode": "max_dt_to_1",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.008,
        "grad_clip_value": 20.0,
    },
    "n87_hardcut_max_step": {
        "material": "N87",
        "paper_name": "HardCutRateGRU",
        "model_key": "SoftplusRateHardcutGRU",
        "dt_norm_mode": "max_dt_to_1",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.004,
        "grad_clip_value": 20.0,
    },
    "n87_hardcut_rms_slope": {
        "material": "N87",
        "paper_name": "HardCutRateGRU",
        "model_key": "SoftplusRateHardcutGRU",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.008,
        "grad_clip_value": 20.0,
    },
    "n87_globalexp_max_step": {
        "material": "N87",
        "paper_name": "GlobalExpRateGRU",
        "model_key": "SoftplusRateGlobalExpGRU",
        "dt_norm_mode": "max_dt_to_1",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.006,
        "grad_clip_value": 20.0,
    },
    "n87_globalexp_rms_slope": {
        "material": "N87",
        "paper_name": "GlobalExpRateGRU",
        "model_key": "SoftplusRateGlobalExpGRU",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": 0.001,
        "lr_3": 0.008,
        "grad_clip_value": 20.0,
    },
}


TEN_MATERIAL_SETTINGS = {
    "N87": (0.001, 0.006, 20.0),
    "N49": (0.002, 0.006, 20.0),
    "3C94": (0.001, 0.004, 20.0),
    "3E6": (0.001, 0.008, 20.0),
    "78": (0.001, 0.004, 20.0),
    "3F4": (0.001, 0.006, 20.0),
    "77": (0.001, 0.008, 20.0),
    "N30": (0.001, 0.006, 20.0),
    "N27": (0.001, 0.006, 20.0),
    "3C90": (0.001, 0.004, 20.0),
}


TEN_MATERIAL_EXPERIMENTS = {
    f"material_{material}": {
        "material": material,
        "paper_name": "RateGRU",
        "model_key": "SoftplusRateExpCapGRU",
        "dt_norm_mode": "std_db_dt",
        "max_len": 4000,
        "lr_12": lr_12,
        "lr_3": lr_3,
        "grad_clip_value": clip,
    }
    for material, (lr_12, lr_3, clip) in TEN_MATERIAL_SETTINGS.items()
}


PAPER_EXPERIMENTS = {**N87_EXPERIMENTS, **TEN_MATERIAL_EXPERIMENTS}


def get_paper_experiment(experiment_id):
    try:
        return PAPER_EXPERIMENTS[experiment_id].copy()
    except KeyError as exc:
        valid = ", ".join(PAPER_EXPERIMENTS)
        raise ValueError(f"Unknown experiment '{experiment_id}'. Valid values: {valid}") from exc
