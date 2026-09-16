import torch
from rl_100.model.diffusion.ema_model import EMAModel as OriginalEMA


class EMAModel(OriginalEMA):
    """Preserve upstream parameter averaging and synchronize model buffers."""
    @torch.no_grad()
    def step(self, new_model):
        super().step(new_model)
        destination = dict(self.averaged_model.named_buffers())
        for name, buffer in new_model.named_buffers():
            destination[name].copy_(buffer)


if __name__ == '__main__':
    import copy
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    ema = EMAModel(copy.deepcopy(model))
    ema.optimization_step = 200
    for i in range(3):
        model(torch.randn(16, 4) + 3 * i)
        ema.step(model)
        assert all(torch.equal(b, dict(ema.averaged_model.named_buffers())[k])
                   for k, b in model.named_buffers())
    assert ema.optimization_step == 203
    assert ema.averaged_model[1].num_batches_tracked.item() == 3
    assert torch.isfinite(ema.averaged_model(torch.randn(16, 4))).all()
    print('EMA_BUFFER_REGRESSION_PASS')
