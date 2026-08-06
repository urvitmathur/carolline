"""Shared MuJoCo passive-viewer loop with batched physics and FPS pacing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import mujoco


def tune_viewer_for_speed(
    viewer: Any,
    *,
    keep_textures: bool = False,
    keep_lights: bool = False,
) -> None:
    """Disable expensive debug overlays in the passive viewer."""
    opt = viewer.opt
    skip: set[Any] = set()
    if keep_textures:
        skip.add(mujoco.mjtVisFlag.mjVIS_TEXTURE)
    if keep_lights:
        skip.add(mujoco.mjtVisFlag.mjVIS_LIGHT)
    for flag in (
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
        mujoco.mjtVisFlag.mjVIS_JOINT,
        mujoco.mjtVisFlag.mjVIS_CAMERA,
        mujoco.mjtVisFlag.mjVIS_LIGHT,
        mujoco.mjtVisFlag.mjVIS_TEXTURE,
        mujoco.mjtVisFlag.mjVIS_CONVEXHULL,
        mujoco.mjtVisFlag.mjVIS_FLEXVERT,
        mujoco.mjtVisFlag.mjVIS_FLEXEDGE,
        mujoco.mjtVisFlag.mjVIS_FLEXFACE,
        mujoco.mjtVisFlag.mjVIS_ISLAND,
    ):
        if flag in skip:
            continue
        opt.flags[flag] = False


def run_passive_viewer_loop(
    viewer: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step_fn: Callable[[], bool],
    *,
    substeps: int = 20,
    target_fps: float = 30.0,
    realtime: bool = False,
    realtime_speed: float = 1.0,
    sync_state_only: bool = True,
    control_every: int | None = None,
    control_once_per_frame: bool = True,
    on_frame: Callable[[Any, mujoco.MjData], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    print_fps: bool = True,
    hud_every: int = 1,
) -> None:
    """Run physics in batches per visual frame for smoother viewer playback.

    By default control runs **once per visual frame** and the same motor
    command is held for all ``substeps`` physics integrations — this is the
    main lever for interactive FPS.
    """
    substeps = max(1, int(substeps))
    if control_every is None:
        control_stride = substeps if control_once_per_frame else 1
    else:
        control_stride = max(1, int(control_every))
    target_fps = max(1.0, float(target_fps))
    realtime_speed = max(0.05, float(realtime_speed))
    frame_budget = 1.0 / target_fps
    hud_every = max(1, int(hud_every))

    wall_t0 = time.perf_counter()
    sim_t0 = data.time
    fps_frames = 0
    fps_window_t0 = wall_t0
    hud_counter = 0

    while viewer.is_running():
        if should_continue is not None and not should_continue():
            break

        frame_start = time.perf_counter()
        running = True

        for sub in range(substeps):
            if should_continue is not None and not should_continue():
                running = False
                break
            if sub % control_stride == 0:
                running = step_fn()
                if not running:
                    break
            mujoco.mj_step(model, data)

        hud_counter += 1
        if on_frame is not None and hud_counter >= hud_every:
            on_frame(viewer, data)
            hud_counter = 0

        viewer.sync(state_only=sync_state_only)

        fps_frames += 1
        if print_fps and fps_frames >= int(max(target_fps, 1.0)):
            now = time.perf_counter()
            window = now - fps_window_t0
            if window > 0.0:
                print(f"Viewer: {fps_frames / window:.0f} FPS  sim_t={data.time:.2f}s", flush=True)
            fps_frames = 0
            fps_window_t0 = now

        if not running:
            break

        if realtime:
            sim_elapsed = data.time - sim_t0
            wall_elapsed = time.perf_counter() - wall_t0
            delay = (sim_elapsed / realtime_speed) - wall_elapsed
            if delay > 0.0:
                time.sleep(min(delay, frame_budget))
        else:
            elapsed = time.perf_counter() - frame_start
            sleep_s = frame_budget - elapsed
            if sleep_s > 0.0:
                time.sleep(sleep_s)
