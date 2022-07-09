from pyparsing import indentedBlock
import torch
from pykeops.torch import LazyTensor
from torch.nn import Module, ModuleList, Sequential
from torch import nn
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv, GATv2Conv

#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])) 


class DGM_d(nn.Module):
    def __init__(self, hidden_dim, k=5, sparse=True):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(4.).float())
        self.linear_proj = MLP(hidden_dim)
        self.k = k
        
    def forward(self, x):

        x = self.linear_proj(x)
        edges_hat, logprobs = self.sample_without_replacement(x)
                
        return edges_hat, logprobs
    
    def sample_without_replacement(self, x):
        
        b,n,_ = x.shape
        
        G_i = LazyTensor(x[:, :, None, :])    # (batch, n, 1, d)
        X_j = LazyTensor(x[:, None, :, :])    # (batch, 1, n, d)
    
        distance_node = ((G_i - X_j) ** 2).sum(-1) # (batch, n, n) this is the distance between each node

        # argKmin already add gumbel noise
        distance_temperature = distance_node * torch.exp(torch.clamp(self.temperature,-5,5))
        indices = distance_temperature.argKmin(self.k, dim=1)

        # compute the logprobs
        x1 = torch.gather(x, -2, indices.view(indices.shape[0],-1)[...,None].repeat(1,1,x.shape[-1]))
        x2 = x[:,:,None,:].repeat(1,1,self.k,1).view(x.shape[0],-1,x.shape[-1])

        distance_node_graph = ((x1 - x2) ** 2).sum(-1)
        logprobs = (-distance_node_graph * torch.exp(torch.clamp(self.temperature, -5, 5))).reshape(x.shape[0], -1, self.k)

        rows = torch.arange(n).view(1, n, 1).to(x.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b,-1), rows.view(b,-1)), -1)

        return edges, logprobs
 
class MLP(nn.Module): 
    def __init__(self, hidden_dim=64):
        super(MLP, self).__init__()

        self.MLP = nn.Sequential(
         nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(), 
         nn.Linear(hidden_dim, hidden_dim), 
         nn.ReLU(),
         nn.Linear(hidden_dim, hidden_dim)
         )
            
    def forward(self, x):
        return self.MLP(x)
        
    
class Identity(nn.Module):
    def __init__(self,retparam=None):
        self.retparam=retparam
        super(Identity, self).__init__()
        
    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params
    