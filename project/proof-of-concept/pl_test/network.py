import time
import torch

import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from dgl.nn import GraphConv
from data import dict2dgl, task_loss, task_corr


class dict_model(pl.LightningModule):
    def __init__(self, features_dim=23, label_type='lddt', time_limit=0, start_time=None):
        super().__init__()

        self.fc_test = nn.Linear(features_dim, 1)
        self.label_type = label_type
        if start_time is None:
            self.time_start = time.time()
        else:
            self.time_start = start_time
        self.time_limit = time_limit

    def network_step(self, g):
        one_hot = g['one_hot'][0]
        features = g['features'][0]
        score_mask = g['lddt_mask'][0]
        x = torch.cat([one_hot, features], dim=1).type_as(one_hot)
        x = self.fc_test(x)
        score_mask = torch.unsqueeze(score_mask, 1)
        x = torch.mul(x, score_mask)
        return x.unsqueeze(0)

    def forward(self, x):
        y_hat = self.network_step(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x = batch
        pred = self.network_step(x).squeeze()
        y = x[self.label_type].squeeze()
        x_mask_np = x['lddt_mask'].squeeze()
        loss, _ = task_loss(pred[x_mask_np], y[x_mask_np])
        corr = task_corr(pred[x_mask_np], y[x_mask_np])
        batch_dictionary = {'loss': loss,
                            'corr': corr
                            }
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('train_corr', corr, on_epoch=True, sync_dist=True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x = batch
        pred = self.network_step(x).squeeze()
        y = x[self.label_type].squeeze()
        x_mask_np = x['lddt_mask'].squeeze()
        loss, _ = task_loss(pred[x_mask_np], y[x_mask_np])
        corr = task_corr(pred[x_mask_np], y[x_mask_np])
        batch_dictionary = {'val_loss': loss,
                            'val_corr': corr
                            }
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_corr', corr, on_epoch=True, sync_dist=True)
        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_epoch_end(self):
        time_elapsed = time.time() - self.time_start
        time_left = self.time_limit - time_elapsed
        avg_epoch_time = time_elapsed / (self.current_epoch + 1)
        if self.time_limit > 0 and time_left < avg_epoch_time * 1.5:
            print('At epoch: {}'.format(self.current_epoch))
            print('Time left: {}s less than 1.5 times of average epoch time: {}s, exiting...'.format(time_left,
                                                                                                     avg_epoch_time))
            exit()


class graph_model(pl.LightningModule):
    def __init__(self, features_dim=23, label_type='lddt', time_limit=0, start_time=None):
        super().__init__()

        self.fc_test = nn.Linear(features_dim, 1)
        self.label_type = label_type
        if start_time is None:
            self.time_start = time.time()
        else:
            self.time_start = start_time
        self.time_limit = time_limit
        self.conv1 = GraphConv(23, 4)
        self.conv2 = GraphConv(4, 1)

    def network_step(self, g):
        g1 = dict2dgl(g)
        h = g1.ndata['f'].squeeze()
        score_mask = g['lddt_mask'].squeeze()

        h = self.conv1(g1, h)
        h = F.relu(h)
        h = self.conv2(g1, h)
        x = torch.mul(h.squeeze(), score_mask)
        return x.squeeze(0)

    def forward(self, x):
        y_hat = self.network_step(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x = batch
        pred = self.network_step(x).squeeze()
        y = x[self.label_type].squeeze()
        x_mask_np = x['lddt_mask'].squeeze()
        loss, _ = task_loss(pred[x_mask_np], y[x_mask_np])
        corr = task_corr(pred[x_mask_np], y[x_mask_np])
        batch_dictionary = {'loss': loss,
                            'train_corr': corr
                            }
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('train_corr', corr, on_epoch=True, sync_dist=True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x = batch
        pred = self.network_step(x).squeeze()
        y = x[self.label_type].squeeze()
        x_mask_np = x['lddt_mask'].squeeze()
        loss, _ = task_loss(pred[x_mask_np], y[x_mask_np])
        corr = task_corr(pred[x_mask_np], y[x_mask_np])
        batch_dictionary = {'val_loss': loss,
                            'val_corr': corr
                            }
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_corr', corr, on_epoch=True, sync_dist=True)
        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_epoch_end(self):
        time_elapsed = time.time() - self.time_start
        time_left = self.time_limit - time_elapsed
        avg_epoch_time = time_elapsed / (self.current_epoch + 1)
        if self.time_limit > 0 and time_left < avg_epoch_time * 1.5:
            print('At epoch: {}'.format(self.current_epoch))
            print('Time left: {}s less than 1.5 times of average epoch time: {}s, exiting...'.format(time_left,
                                                                                                     avg_epoch_time))
            exit()
