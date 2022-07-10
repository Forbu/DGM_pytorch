import os
import torch
import numpy as np

import torch_geometric as pyg
from torch import nn
from torch.nn import Module, ModuleList, Sequential
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv, GATv2Conv
from torch.utils.data import DataLoader
import torch_scatter

import pytorch_lightning as pl
from argparse import Namespace

from DGMlib.layers import DGM_d

    
class DGM_Model(nn.Module):
    def __init__(self, nb_layer, output_dim, hidden_dim, input_dim, spatial_dim, k=5):
        super(DGM_Model,self).__init__()
        
        self.nb_layer = nb_layer
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim

        # we create the graph generation layer : this module take an input and output a graph
        self.graph_generation = ModuleList()

        # we use the DGM_d class to create the graph generation layer
        for i in range(nb_layer):
            self.graph_generation.append(DGM_d(hidden_dim, k=k, sparse=True))

        # init the graph neural network model
        self.node_preprocessing = nn.ModuleList()

        # create nb_layer of GATv2Conv layers inside the node_preprocessing module
        for i in range(nb_layer):
            #self.node_preprocessing.append(GATv2Conv(input_dim, hidden_dim, heads=1, concat=True))
            self.node_preprocessing.append(GCNConv(hidden_dim, hidden_dim, improved=True, cached=True, add_self_loops=False))
        
        # create the first and the last layer of the graph neural network model with simple linear model
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.last_layer = nn.Linear(hidden_dim, output_dim)

        # layer to preprocess the spatial input
        self.spatial_preprocessing = nn.Linear(spatial_dim, hidden_dim)

    def forward(self, x, x_spatial=None, graph=None, training=True):
        """
        In the forward loop we :
        1. generate the nb_layer graph with the graph generation module using only the spatial input
        2. preprocess the node input with the node preprocessing module
        """
        # pass the concatenated input to the first layer
        x = self.first_layer(x)

        if training:
            # spatial input preprocessing
            x_spatial = self.spatial_preprocessing(x_spatial)

            # generate the nb_layer graph with the graph generation module using only the spatial input
            index_edge = []
            logprobs = []
            for i in range(self.nb_layer):
                index_edge_tmp, logprobs_tmp = self.graph_generation[i](x_spatial)
                index_edge.append(index_edge_tmp.squeeze(0))
                logprobs.append(logprobs_tmp)

            # preprocess the node input with the node preprocessing module
            for i in range(self.nb_layer):
                x = self.node_preprocessing[i](x, index_edge[i].T)

            # pass the output of the first layer to the last layer
            x = self.last_layer(x)

            return x, logprobs, index_edge
        else:
            # generate the nb_layer graph with the graph generation module using only the spatial input
            index_edge = graph

            # preprocess the node input with the node preprocessing module
            for i in range(self.nb_layer):
                x = self.node_preprocessing[i](x, index_edge[i])

            # pass the output of the first layer to the last layer
            x = self.last_layer(x)

            return x




        
