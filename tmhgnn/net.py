# The `TM_HGNN` class implements a graph neural network model with projection from an input feature
# dimension of 768 to 256, utilizing GCN layers for different types of hyperedges and a readout layer
# for graph-level classification.
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, Linear, ReLU


###################    
##### TM-HGNN #####   
###################

"""

class TM_HGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):  
        super(TM_HGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_mask, batch):
        # All hyperedges
        x = self.conv1(x, edge_index)
        x = x.relu()
        # Note hyperedges
        idx_1 = torch.where(edge_mask==1)[0]
        x = self.conv2(x, edge_index[:, idx_1])   
        x = x.relu()
        # Taxonomy hyperedges
        idx_2 = torch.where(edge_mask==2)[0]
        x = self.conv3(x, edge_index[:, idx_2])        

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        
        return x
    

if __name__ == '__main__':
    print("TM-HGNN")

"""

class TM_HGNN(torch.nn.Module):
    """
    TM-HGNN with:
    - 입력: num_features = 4 + emb_dim (clinicalBERT면 4 + 768)
    - 내부에서 emb_dim -> proj_dim (기본 256) 으로 projection
    - meta 정보(앞 4차원)는 그대로 두고, proj된 임베딩과 concat해서 GCN에 입력
    """

    def __init__(self, num_features, hidden_channels, proj_dim=256, dropout=0.3):
        """
        Args:
            num_features: 전체 노드 feature 차원 (4 + emb_dim)
            hidden_channels: GCNConv hidden dim (train.py에서 hidden_channels*heads_1로 들어옴)
            proj_dim: 임베딩 차원 축소 후 차원 (기본 256)
            dropout: 드롭아웃 비율
        """
        super(TM_HGNN, self).__init__()

        # meta(4) + embedding(num_features - 4)
        self.meta_dim = 4
        self.emb_dim = num_features - self.meta_dim   # clinicalBERT면 768, word2vec이면 100
        self.proj_dim = proj_dim
        self.dropout = dropout

        # 768 -> 256 같은 projection 레이어
        self.proj = Linear(self.emb_dim, self.proj_dim)

        # GCN 입력 차원 = meta(4) + proj_dim(256) = 260
        conv_in_dim = self.meta_dim + self.proj_dim

        self.conv1 = GCNConv(conv_in_dim, hidden_channels)
        self.bn1   = BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2   = BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3   = BatchNorm1d(hidden_channels)

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_mask, batch):
        """
        Args:
            x: [num_nodes, 4 + emb_dim]  (train.py에서 data.x_n이 들어옴)
            edge_index: [2, num_edges]
            edge_mask: 각 edge가 어떤 타입 hyperedge인지 (0: word?, 1: note, 2: taxonomy 등)
            batch: [num_nodes] 그래프 배치 인덱스
        """

        # 1) meta / 임베딩 분리
        x_meta = x[:, :self.meta_dim]          # [N, 4]
        x_emb  = x[:, self.meta_dim:]         # [N, emb_dim]

        # 2) emb_dim -> proj_dim
        x_emb_proj = self.proj(x_emb)         # [N, proj_dim]

        # 3) 다시 concat 해서 GCN 입력 만들기
        x = torch.cat([x_meta, x_emb_proj], dim=-1)  # [N, 4 + proj_dim]

        # ---------------- GCN layers ----------------
        # All hyperedges
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Note hyperedges만 사용하는 edge index
        idx_1 = torch.where(edge_mask == 1)[0]
        if idx_1.numel() > 0:
            x = self.conv2(x, edge_index[:, idx_1])
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Taxonomy hyperedges만 사용하는 edge index
        idx_2 = torch.where(edge_mask == 2)[0]
        if idx_2.numel() > 0:
            x = self.conv3(x, edge_index[:, idx_2])
            x = self.bn3(x)
            x = F.relu(x)
            # 마지막 레이어는 dropout 한 번 더 줄지 말지는 취향인데,
            # 과적합이 보이면 여기에도 넣어도 됨.
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout layer: 그래프 단위로 pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)  # [batch_size, 1]

        return x


if __name__ == '__main__':
    print("TM-HGNN with projection (768 -> 256)")