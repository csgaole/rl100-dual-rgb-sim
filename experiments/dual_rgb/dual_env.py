import numpy as np
from gym import spaces

from paired_env import PairedPegEnv


class DualViewPegEnv(PairedPegEnv):
    """Peg insertion with synchronized corner2 and gripperPOV RGB views."""

    camera_keys = {
        "image_corner2": "corner2",
        "image_behindGripper": "behindGripper",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict({
            "image_corner2": spaces.Box(0, 255, (3, self.image_size, self.image_size), dtype=np.uint8),
            "image_behindGripper": spaces.Box(0, 255, (3, self.image_size, self.image_size), dtype=np.uint8),
            "agent_pos": spaces.Box(-np.inf, np.inf, (self.obs_sensor_dim,), dtype=np.float32),
            "full_state": spaces.Box(-np.inf, np.inf, (39,), dtype=np.float64),
        })

    def _render_rgb(self, camera_name):
        # The first render creates the offscreen context. Each subsequent render
        # must explicitly bind this simulator's context and framebuffer because
        # several environments can alternate on one GPU.
        context = self.env.sim._render_context_offscreen
        if context is not None:
            from mujoco_py import const, functions

            context.opengl_context.make_context_current()
            functions.mjr_setBuffer(const.FB_OFFSCREEN, context.con)
        image = self.env.sim.render(
            width=self.image_size,
            height=self.image_size,
            camera_name=camera_name,
            device_id=self.device_id,
        )
        return np.asarray(image).copy()

    def _observe(self, raw):
        # PairedPegEnv renders corner2 once to produce its registered XYZ cloud.
        base = super()._observe(raw)
        corner = np.asarray(base["image"]).copy()
        wrist = self._render_rgb("behindGripper").transpose(2, 0, 1).copy()
        assert corner.shape == wrist.shape == (3, self.image_size, self.image_size)
        assert np.isfinite(corner).all() and np.isfinite(wrist).all()
        return {
            "image_corner2": corner,
            "image_behindGripper": wrist,
            "agent_pos": base["agent_pos"],
            "full_state": base["full_state"],
        }

    def get_rgb(self):
        return {
            "corner2": self._render_rgb("corner2"),
            "behindGripper": self._render_rgb("behindGripper"),
        }
