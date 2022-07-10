import os
import numpy as np
import torch
import pytorch_lightning as pl
from argparse import Namespace
from DGMlib.layers import MLP, DGM_d
from DGMlib.model_dDGM import DGM_Model


class DGM_Model_Training(pl.LightningModule):
    def __init__(self, nb_layer, output_dim, hidden_dim, input_dim, spatial_dim, k=5, lr=0.001):
        super(DGM_Model_Training,self).__init__()

        self.nb_layer = nb_layer
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        self.k = k
        self.lr = lr

        self.model = DGM_Model(nb_layer, output_dim, hidden_dim, input_dim, spatial_dim, k=k)

    def forward(self, x, x_spatial=None, graph=None, training=True):
        return self.model(x, x_spatial, graph, training)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        input = batch['input']
        target = batch['target']
        x_spatial = batch['x_spatial']

        pred, logprobs = self(input, x_spatial, training=True)

        loss_target = torch.nn.functional.mse_loss(pred, target)

        # graph loss to enable the graph regularization
        for i in range(self.nb_layer):
            error_node = (pred - target)**2
            loss_graph = -torch.sum(logprobs[i], axis=-1) * error_node
            loss_target += loss_graph

        self.log('train_loss', loss_target, prog_bar=True)
        self.log('train_loss_graph', loss_graph, prog_bar=True)

        # total loss
        return loss_target + loss_graph


