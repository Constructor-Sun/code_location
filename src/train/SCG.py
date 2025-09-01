import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree

class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super(SGC, self).__init__()
        self.K = K  # 卷积阶数
        self.linear = nn.Linear(in_channels, out_channels)  # 单一线性层

    def forward(self, x, edge_index):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算度矩阵 D
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 归一化邻接矩阵：D^-1/2 * A * D^-1/2
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_weight = norm

        # 预计算 K 次邻接矩阵传播
        h = x
        for _ in range(self.K):
            h = torch.sparse.mm(
                torch.sparse_coo_tensor(edge_index, edge_weight, (x.size(0), x.size(0))),
                h
            )

        # 应用线性变换
        out = self.linear(h)
        return out

# 示例：加载 Cora 数据集并训练 SGC
def train_sgc():
    # 加载数据集
    dataset = Planetoid(root='.', name='Cora')
    data = dataset[0]

    # 初始化模型
    model = SGC(in_channels=dataset.num_features, out_channels=dataset.num_classes, K=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=5e-6)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # 测试
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')

if __name__ == '__main__':
    train_sgc()