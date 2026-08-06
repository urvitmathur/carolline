"""Offscreen MuJoCo renderer for course simulation video capture."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class MujocoVideoRecorder:
    """Capture MuJoCo frames to MP4 using an offscreen renderer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: str | Path,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        if model.vis.global_.offwidth < width or model.vis.global_.offheight < height:
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.azimuth = 130.0
        self._camera.elevation = -22.0
        self._camera.distance = 8.0
        self._camera.lookat[:] = np.array([0.0, 0.0, 0.35], dtype=float)
        self._writer = None
        self._frame_count = 0

    def sync_camera_from_viewer(self, viewer_cam: mujoco.MjvCamera) -> None:
        self._camera.type = viewer_cam.type
        self._camera.fixedcamid = viewer_cam.fixedcamid
        self._camera.trackbodyid = viewer_cam.trackbodyid
        self._camera.lookat[:] = viewer_cam.lookat[:]
        self._camera.distance = viewer_cam.distance
        self._camera.azimuth = viewer_cam.azimuth
        self._camera.elevation = viewer_cam.elevation

    def update_tracking_camera(
        self,
        position: np.ndarray,
        *,
        look_height: float = 0.35,
        distance: float = 8.0,
        azimuth: float = 130.0,
        elevation: float = -22.0,
    ) -> None:
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[0] = float(position[0])
        self._camera.lookat[1] = float(position[1])
        self._camera.lookat[2] = float(position[2]) + look_height
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation

    def capture(self, data: mujoco.MjData) -> None:
        self.renderer.update_scene(data, camera=self._camera)
        frame = self.renderer.render()
        if self._writer is None:
            import imageio

            self._writer = imageio.get_writer(
                str(self.output_path),
                fps=self.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=1,
            )
        self._writer.append_data(frame)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def frame_count(self) -> int:
        return self._frame_count
