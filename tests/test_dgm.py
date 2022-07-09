
from DGMlib.layers import DGM_d
from DGMlib.model_dDGM import DGM_Model
import torch
from prettytable import PrettyTable


from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv, GATv2Conv

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

def test_DGM():
    nb_layer = 3
    output_dim = 2
    hidden_dim = 128
    input_dim = 128
    spatial_dim = 5
    k = 5
    batch_size = 2
    nb_node = 40000
    
    # init layer
    layer = DGM_d(hidden_dim, k=k, sparse=True)
    layer = layer.cuda()
    
    # init tensor
    x = torch.randn(batch_size, nb_node, input_dim)
    x = x.cuda()

    # pass though the layer
    index_edge, logprobs = layer(x)

    # cheching the shape of the output
    assert logprobs.shape == (batch_size, nb_node, k)
    assert index_edge.shape == (batch_size, nb_node* k, 2)

def test_GCN():

    hidden_dim = 128
    input_dim = 128
    k = 4
    batch_size = 2
    nb_node = 40000
    
    # init layer
    layer = GCNConv(input_dim, hidden_dim, improved=True, cached=True, add_self_loops=False)
    layer = layer.cuda()
    
    # init tensor
    x = torch.randn(batch_size, nb_node, input_dim)
    x = x.cuda()

    # index graph
    index_edge = torch.randint(0, nb_node, (2, nb_node* k))
    index_edge = index_edge.cuda()

    # pass though the layer
    result = layer(x, index_edge)

    # cheching the shape of the output
    assert result.shape == (batch_size, nb_node, hidden_dim)

def test_DGM_Model():
    nb_layer = 2
    output_dim = 2
    hidden_dim = 128
    input_dim = 12
    spatial_dim = 2
    k = 5
    batch_size = 2
    nb_node = 40000
    
    # init model
    model = DGM_Model(nb_layer, output_dim, hidden_dim, input_dim, spatial_dim, k=k)
    model = model.cuda()

    # init tensor
    x = torch.randn(batch_size, nb_node, input_dim)
    x = x.cuda()
    x_spatial = torch.randn(1, nb_node, spatial_dim)
    x_spatial = x_spatial.cuda()
    
    # pass though the model
    result, logprobs, index_edge = model(x, x_spatial, training=True)

    # cheching the shape of the output
    assert result.shape == (batch_size, nb_node, output_dim)




    

