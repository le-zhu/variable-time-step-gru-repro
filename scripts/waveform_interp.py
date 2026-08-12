#!/usr/bin/env python3
"""Waveform interpolation used when upsampling B-H sequences onto a finer grid.

Linear interpolation makes the forward difference of the reconstructed B exactly
piecewise constant: within one original interval every increment equals
dB_orig / factor, so dB/dt collapses to the original values each repeated
`factor` times. The resulting staircase is an input the model never sees during
training, and it is what drives the ringing observed in upsampled rollouts.

A C^2 interpolant removes the staircase without touching the forward-difference
convention, which must stay as-is because the model is trained on
(B_{k+1} - B_k) / dt paired with that same dt.

Benchmarked by decimating the 50 kHz group (320 pts/cycle) to 20 pts/cycle and
reconstructing; and end-to-end through the RateGRU rollout on origin_test.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

DEFAULT_METHOD = "cubic"

_BUILDERS = {
    "linear": lambda x, y: (lambda xu: np.interp(xu, x, y)),
    "cubic": lambda x, y: CubicSpline(x, y),
    "akima": lambda x, y: Akima1DInterpolator(x, y),
    "pchip": lambda x, y: PchipInterpolator(x, y),
}

INTERP_METHODS = tuple(_BUILDERS)


def interpolate_channel(original_x, values, upsampled_x, method=DEFAULT_METHOD):
    """Resample `values` sampled at `original_x` onto `upsampled_x`.

    `method` must be one of INTERP_METHODS. Returns float64; callers cast.
    """
    if method not in _BUILDERS:
        raise ValueError(f"unknown interpolation method {method!r}; expected one of {INTERP_METHODS}")
    y = np.asarray(values, dtype=np.float64)
    return np.asarray(_BUILDERS[method](np.asarray(original_x, dtype=np.float64), y)(upsampled_x),
                      dtype=np.float64)
