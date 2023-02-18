# Courtesy of GraphGPS, https://github.com/rampasek/GraphGPS
import torch


class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.lin = torch.nn.Linear(input_channels, output_channels)

    def forward(self, batch):
        batch.x = batch.x.to(torch.float32)
        batch.x = self.lin(batch.x)
        return batch

class LinearEdgeEncoder(torch.nn.Module):
    
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.lin = torch.nn.Linear(input_channels, output_channels)
        
    def forward(self, batch):
        batch.edge_attr = self.lin(batch.edge_attr.type(torch.float32))
        return batch
    
    