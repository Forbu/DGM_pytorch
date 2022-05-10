import os
import torch
import numpy as np

import torch_geometric
from torch import nn
from torch.nn import Module, ModuleList, Sequential
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv
from torch.utils.data import DataLoader
import torch_scatter

import pytorch_lightning as pl
from argparse import Namespace

from DGMlib.layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS")=="False":
    from DGMlib.layers_dense import *

class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model,self).__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

#         self.hparams=hparams
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k


        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in hparams or hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),k=hparams.k,distance=hparams.distance))
#                 self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
            else:
                self.graph_f.append(Identity())

            if hparams.gfun == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == 'gcn':
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'gat':
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))

        self.fc = MLP(fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None


    def forward(self,x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)

        graph_x = x.detach()
        lprobslist = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,lprobs = f(graph_x,edges,None)
            b,n,d = x.shape

#             edges,_ = torch_geometric.utils.remove_self_loops(edges)
#             edges,_ = torch_geometric.utils.add_self_loops(edges)

            self.edges=edges
            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.hparams.dropout, train=self.training), edges)).view(b,n,-1)
            graph_x = torch.cat([graph_x,x.detach()],-1)
            if lprobs is not None:
                lprobslist.append(lprobs)

        return self.fc(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        X, y, _, edges = train_batch
        edges = edges[0]

        pred,logprobs = self(X,edges)

        train_pred = X
        train_lab = y

        loss_target = torch.nn.functional.mse_loss(train_pred, train_lab)

        # graph loss
        point_w = ((train_pred - train_lab)**2 - loss_target).detach()
        graph_loss = point_w * logprobs.exp().mean([-1,-2])

        graph_loss = graph_loss.mean()

        self.log('train_graph_loss', graph_loss.detach().cpu())
        self.log('train_loss', loss_target.detach().cpu())

        return loss_target + graph_loss


