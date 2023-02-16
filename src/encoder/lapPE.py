import torch
import torch_geometric.utils as tg


#from src.GraphGPS.graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder as enc

def lap_pe(dataset, device):
    """
    Computes positional encoding for each node by the eigenvalues of the graph Laplacian matrix.
    """
    data = dataset.data
    data.to(device)
    x = data.x
    edge_index = data.edge_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lap = tg.get_laplacian(edge_index, normalization="sym", num_nodes=x.size(0))[0]
    lap = tg.to_dense_adj(lap)  # , size=x.size(0))
    eigvals = torch.linalg.eigvalsh(lap, UPLO='U')
    eigvals = eigvals.to(device)
    data.x = torch.cat((data.x, eigvals.T), 1)