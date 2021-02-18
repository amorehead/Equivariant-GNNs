import torch
from pytorch_lightning.metrics import Metric

from project.utils.utils import norm2units


class L1Loss(Metric):
    def __init__(self, std, mean, task, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.std = std
        self.mean = mean
        self.task = task

        self.add_state("l1_loss", default=torch.tensor(1), dist_reduce_fx="mean")
        self.add_state("rescaled_l1_loss", default=torch.tensor(1), dist_reduce_fx="mean")

    def _input_format(self, preds, target):
        return torch.squeeze(preds), torch.squeeze(target)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.l1_loss = torch.sum(torch.abs(preds - target))
        self.rescaled_l1_loss = norm2units(self.l1_loss, self.std, self.mean, self.task)

    def compute(self):
        return self.l1_loss, self.rescaled_l1_loss


class L2Loss(Metric):
    def __init__(self, dist_sync_on_step=False, use_mean=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.use_mean = use_mean

        self.add_state("l2_loss", default=torch.tensor(1), dist_reduce_fx="mean")

    def _input_format(self, preds, target):
        return torch.squeeze(preds), torch.squeeze(target)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.l2_loss = torch.sum((preds - target) ** 2)

    def compute(self):
        return self.l2_loss
