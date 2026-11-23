import sys
import pytest
import torch
import networkx as nx

def test_python_version():
    assert sys.version_info >= (3, 10), f"Expected Python >= 3.10, got {sys.version}"

def test_torch_cuda():
    assert torch.__version__ is not None
    print(f"\nPyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        # Run a quick tensor computation on GPU
        x = torch.ones(2, 2, device="cuda")
        y = x + x
        assert torch.all(y == 2)

def test_torch_geometric():
    import torch_geometric
    from torch_geometric.nn import GATConv
    
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    conv = GATConv(in_channels=16, out_channels=32, heads=2)
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)
    out = conv(x, edge_index)
    assert out.shape == (4, 64)  # 32 * 2 heads

def test_networkx_graph_operations():
    G = nx.karate_club_graph()
    assert G.number_of_nodes() == 34
    sp = nx.shortest_path(G, source=0, target=33)
    assert len(sp) >= 2

def test_project_import():
    import src
    assert src.__version__ == "0.1.0"
