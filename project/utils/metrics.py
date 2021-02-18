import torch
from pytorch_lightning.metrics import Metric


class L1L2Loss(Metric):
    def __init__(self, dist_sync_on_step=False, use_mean=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.use_mean = use_mean

        self.add_state("l1_loss", default=torch.tensor(1), dist_reduce_fx="mean")
        self.add_state("l2_loss", default=torch.tensor(1), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # assert preds.shape == target.shape

        self.l1_loss = torch.sum(torch.abs(preds - target))
        self.l2_loss = torch.sum((preds - target) ** 2)

    def compute(self):
        return self.l1_loss, self.l2_loss
