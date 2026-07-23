"""Environment disturbances applied by the validation observer (not the controller)."""

from __future__ import annotations

import numpy as np

from carolline_control.validation.randomization import RunParameters


class DisturbanceManager:
    """Apply wind and gusts before mj_step."""

    def __init__(self, params: RunParameters, body_id: int) -> None:
        self._params = params
        self._body_id = body_id
        self._base_wind = params.wind_vector.copy()
        self._gust_idx = 0
        self._sin_phase = 0.0
        self._mode = "nominal"

    def set_campaign_mode(self, mode: str) -> None:
        self._mode = mode

    def apply(self, model, data, time: float, dt: float) -> np.ndarray:
        wind = self._base_wind.copy()

        if self._mode == "sinusoidal_wind":
            self._sin_phase += dt
            wind[0] += 2.0 * np.sin(0.4 * self._sin_phase)
            wind[1] += 1.5 * np.cos(0.25 * self._sin_phase)

        if self._mode == "dryden_turbulence":
            wind += np.random.default_rng(int(time * 1000) % 2**31).normal(0, 0.8, 3)
            wind[2] *= 0.2

        while (
            self._gust_idx < len(self._params.wind_gust_times)
            and time >= self._params.wind_gust_times[self._gust_idx]
        ):
            wind += self._params.wind_gust_vectors[self._gust_idx]
            self._gust_idx += 1

        model.opt.wind[:] = wind
        data.xfrc_applied[self._body_id, :] = 0.0
        return wind
