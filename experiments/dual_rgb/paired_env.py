import numpy as np
from rl_100.env.metaworld.metaworld_wrapper import MetaWorldEnv
from rl_100.gym_util.mjpc_wrapper import _resolve_render_device_id

class PairedPegEnv(MetaWorldEnv):
    """Pinned v2 wrapper: one frozen simulator state supplies RGB, depth and XYZ."""
    def __init__(self, task_name='peg-insert-side', **kwargs):
        super().__init__(task_name, **kwargs)
        self.device_id = _resolve_render_device_id()
        self._next_seed = 700000
        capture = self.pc_generator.captureImage
        def cached_capture(*args, **kw):
            # MuJoCo does not select this renderer's framebuffer on every render.
            # Bind it explicitly when alternating independent simulation environments.
            context = self.env.sim._render_context_offscreen
            if context is not None:
                from mujoco_py import functions, const
                context.opengl_context.make_context_current()
                functions.mjr_setBuffer(const.FB_OFFSCREEN, context.con)
            rgb, depth = capture(*args, **kw)
            self._paired_rgb = rgb.copy()
            self._paired_depth = depth.copy()
            return rgb, depth
        self.pc_generator.captureImage = cached_capture
        # Context creation forwards MuJoCo and overwrites manual target sites;
        # warm it before the first seeded reset/observation.
        self.pc_generator.captureImage('corner2', device_id=self.device_id)

    def seed(self, seed=None):
        self._next_seed = int(700000 if seed is None else seed)
        return [self._next_seed]

    set_seed = seed

    def _observe(self, raw):
        for site in self.env._target_site_config:
            self.env._set_pos_site(*site)
        pc, _ = self.get_point_cloud(use_rgb=False)
        assert pc.shape == (512, 3) and np.isfinite(pc).all()
        return dict(image=self._paired_rgb.transpose(2,0,1).copy(),
                    depth=self._paired_depth.copy(), point_cloud=pc.astype(np.float32),
                    agent_pos=self.get_robot_state().astype(np.float32),
                    full_state=np.asarray(raw).copy())

    def reset(self):
        self.env.seed(self._next_seed)
        rng = np.random.get_state()
        np.random.seed(self._next_seed)
        self._next_seed += 1
        self.env._freeze_rand_vec = False
        try:
            raw = self.env.reset()
            # reset_model changes model.body_pos after its last forward call.
            # Synchronize geometry before rendering the initial observation.
            self.env.sim.forward()
            # Frame stacking must not retain the previous episode's state.
            raw = self.env._get_obs()
            self.env._prev_obs = raw[:18].copy()
            raw = self.env._get_obs()
        finally:
            np.random.set_state(rng)
        self.cur_step = 0
        return self._observe(raw)

    def step(self, action):
        raw, reward, terminated, info = self.env.step(action)
        self.cur_step += 1
        timeout = self.cur_step >= self.episode_length and not terminated
        info = dict(info, **{'TimeLimit.truncated': bool(timeout)})
        obs = self._observe(raw)
        return obs, reward, bool(terminated or timeout), info

    def get_visual_obs(self):
        return self._observe(self.env._get_obs())

    def get_rgb(self):
        return self._paired_rgb.copy()
