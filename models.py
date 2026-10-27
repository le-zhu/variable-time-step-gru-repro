from dataclasses import dataclass

import torch
import torch.nn as nn


class DtScaledGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.gru_wz = nn.Linear(4, self.hidden_size)
        self.gru_uz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wr = nn.Linear(4, self.hidden_size)
        self.gru_ur = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wh = nn.Linear(4, self.hidden_size)
        self.gru_uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward_step(self, input, h, hs, dt):
        gru_input = torch.cat([input, h], dim=1)
        update_coeff = torch.sigmoid(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        a_t = update_coeff * dt
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next


class StandardGRU(DtScaledGRU):
    """Fixed-step comparison branch using the direct candidate-update convention."""

    def forward_step(self, input, h, hs, dt):
        gru_input = torch.cat([input, h], dim=1)
        update_coeff = torch.sigmoid(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        a_t = update_coeff
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next


class DtInputFixedGRU(DtScaledGRU):
    """Fixed-update GRU with normalized dt provided as a fifth input feature.

    This is DtInputGRU in the paper. It uses the same hand-written reset-before
    cell and initialization as the other models; dt enters only the gate and
    candidate inputs, not the update.
    """

    def __init__(self):
        super().__init__()
        self.gru_wz = nn.Linear(5, self.hidden_size)
        self.gru_wr = nn.Linear(5, self.hidden_size)
        self.gru_wh = nn.Linear(5, self.hidden_size)

    def forward_step(self, input, h, hs, dt):
        if dt.dim() == 1:
            dt = dt.unsqueeze(1)
        dt = dt.to(device=input.device, dtype=input.dtype)
        gru_input = torch.cat([input, h, dt], dim=1)
        update_coeff = torch.sigmoid(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        hs_next = hs + update_coeff * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next


class SoftplusRateSoftcutGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.gru_wz = nn.Linear(4, self.hidden_size)
        self.gru_uz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wr = nn.Linear(4, self.hidden_size)
        self.gru_ur = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wh = nn.Linear(4, self.hidden_size)
        self.gru_uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward_step(self, input, h, hs, dt):
        gru_input = torch.cat([input, h], dim=1)
        z_t = torch.nn.functional.softplus(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        z_dt = z_t * dt
        a_t = 0.95 * torch.clamp(z_dt, min=0.0, max=1.0) + 0.05 * torch.tanh(z_dt)
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next


class SoftplusRateHardcutGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.gru_wz = nn.Linear(4, self.hidden_size)
        self.gru_uz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wr = nn.Linear(4, self.hidden_size)
        self.gru_ur = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wh = nn.Linear(4, self.hidden_size)
        self.gru_uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward_step_with_diagnostics(self, input, h, hs, dt):
        gru_input = torch.cat([input, h], dim=1)
        z_t = torch.nn.functional.softplus(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        z_dt = z_t * dt
        a_t = torch.clamp(z_dt, min=0.0, max=1.0)
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next, {"sdt": z_dt}

    def forward_step(self, input, h, hs, dt):
        h_next, hs_next, _diagnostics = self.forward_step_with_diagnostics(input, h, hs, dt)
        return h_next, hs_next


class SoftplusRateExpCapGRU(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_wz = nn.Linear(4, self.hidden_size)
        self.gru_uz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wr = nn.Linear(4, self.hidden_size)
        self.gru_ur = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wh = nn.Linear(4, self.hidden_size)
        self.gru_uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward_step(self, input, h, hs, dt):
        gru_input = torch.cat([input, h], dim=1)
        z_t = torch.nn.functional.softplus(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        z_dt = z_t * dt
        a_t = torch.where(
            z_dt <= 0.8,
            z_dt,
            1.0 - 0.2 * torch.exp(-(z_dt - 0.8) / 0.2),
        )
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next


class SoftplusRateGlobalExpGRU(SoftplusRateExpCapGRU):
    """RateGRU with the global exponential update map ``1 - exp(-s * dt)``."""

    def forward_step_with_diagnostics(self, input, h, hs, dt):
        gru_input = torch.cat([input, h], dim=1)
        z_t = torch.nn.functional.softplus(self.gru_wz(gru_input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(gru_input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(gru_input) + self.gru_uh(r_t * hs))
        z_dt = z_t * dt
        a_t = -torch.expm1(-z_dt)
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next, {"sdt": z_dt}

    def forward_step(self, input, h, hs, dt):
        h_next, hs_next, _diagnostics = self.forward_step_with_diagnostics(input, h, hs, dt)
        return h_next, hs_next


class SoftplusRateExpCapGRUHidden32(SoftplusRateExpCapGRU):
    def __init__(self):
        super().__init__(hidden_size=32)


class SoftplusRateExpCapGRUNoH(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_wz = nn.Linear(3, self.hidden_size)
        self.gru_uz = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wr = nn.Linear(3, self.hidden_size)
        self.gru_ur = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru_wh = nn.Linear(3, self.hidden_size)
        self.gru_uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward_step(self, input, h, hs, dt):
        z_t = torch.nn.functional.softplus(self.gru_wz(input) + self.gru_uz(hs))
        r_t = torch.sigmoid(self.gru_wr(input) + self.gru_ur(hs))
        hs_candidate = torch.tanh(self.gru_wh(input) + self.gru_uh(r_t * hs))
        z_dt = z_t * dt
        a_t = torch.where(
            z_dt <= 0.8,
            z_dt,
            1.0 - 0.2 * torch.exp(-(z_dt - 0.8) / 0.2),
        )
        hs_next = hs + a_t * (hs_candidate - hs)
        h_next = self.head(hs_next)
        return h_next, hs_next


@dataclass(frozen=True)
class ModelEntry:
    label: str
    model_class: type
    default_run_dir: str
    default_dt_norm_mode: str


MODEL_REGISTRY = {
    "StandardGRU": ModelEntry(
        label="StandardGRU",
        model_class=StandardGRU,
        default_run_dir="runs/StandardGRU",
        default_dt_norm_mode="max_dt_to_1",
    ),
    "DtScaledGRU": ModelEntry(
        label="DtScaledGRU",
        model_class=DtScaledGRU,
        default_run_dir="runs/DtScaledGRU",
        default_dt_norm_mode="max_dt_to_1",
    ),
    "DtInputFixedGRU": ModelEntry(
        label="DtInputFixedGRU",
        model_class=DtInputFixedGRU,
        default_run_dir="runs/DtInputFixedGRU",
        default_dt_norm_mode="std_db_dt",
    ),
    "SoftplusRateSoftcutGRU": ModelEntry(
        label="SoftplusRateSoftcutGRU",
        model_class=SoftplusRateSoftcutGRU,
        default_run_dir="runs/SoftplusRateSoftcutGRU",
        default_dt_norm_mode="std_db_dt",
    ),
    "SoftplusRateHardcutGRU": ModelEntry(
        label="SoftplusRateHardcutGRU",
        model_class=SoftplusRateHardcutGRU,
        default_run_dir="runs/SoftplusRateHardcutGRU",
        default_dt_norm_mode="std_db_dt",
    ),
    "SoftplusRateExpCapGRU": ModelEntry(
        label="SoftplusRateExpCapGRU",
        model_class=SoftplusRateExpCapGRU,
        default_run_dir="runs/SoftplusRateExpCapGRU",
        default_dt_norm_mode="std_db_dt",
    ),
    "SoftplusRateGlobalExpGRU": ModelEntry(
        label="SoftplusRateGlobalExpGRU",
        model_class=SoftplusRateGlobalExpGRU,
        default_run_dir="runs/SoftplusRateGlobalExpGRU",
        default_dt_norm_mode="std_db_dt",
    ),
    "SoftplusRateExpCapGRUHidden32": ModelEntry(
        label="SoftplusRateExpCapGRUHidden32",
        model_class=SoftplusRateExpCapGRUHidden32,
        default_run_dir="runs/SoftplusRateExpCapGRUHidden32",
        default_dt_norm_mode="std_db_dt",
    ),
    "SoftplusRateExpCapGRUNoH": ModelEntry(
        label="SoftplusRateExpCapGRUNoH",
        model_class=SoftplusRateExpCapGRUNoH,
        default_run_dir="runs/SoftplusRateExpCapGRUNoH",
        default_dt_norm_mode="std_db_dt",
    ),
}


def get_model_entry(model_key):
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_key}'. Valid models: {valid}") from exc


def get_model_keys():
    return list(MODEL_REGISTRY.keys())
