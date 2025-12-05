
import torch.nn as nn

from ablation_mpnn import MPNNGNN
from ablation_Feature_Fusion import Fusion
import torch

class MultiHeadAttentionClassifier(nn.Module):
    def __init__(self, input_dim=64, d_model=128, nhead=4, num_classes=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        """
        x = self.input_proj(x)  # 投影到模型维度
        attn_output, _ = self.attn(x, x, x)  # 自注意力
        x = attn_output.mean(dim=1)  # 池化（平均所有位置）
        out = self.fc(x)  # 分类
        return out

class MPNNPredictorWithProtein(nn.Module):
    def __init__(self,
                 node_in_feats=74,  # 节点特征维度
                 edge_in_feats=12,  # 边特征维度
                 protein_feats=1280,  #蛋白特征维度     ESM-2输出的特征维度1280
                 node_out_feats=64,     #default=64
                 edge_hidden_feats=128,     #default=128
                 n_tasks=1,     #default=1
                 num_step_message_passing=6,        #default=6
                 num_step_set2set=6,        #default=6
                 num_layer_set2set=3,        #default=3
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super().__init__()
        self.fusion = Fusion().to(device)

        # MPNN部分：处理化合物的图结构
        '''输出维度是输出维度是node_out_feats'''
        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing,
                           )


        #蛋白质特征处理
        self.protein_projector = nn.Sequential(
            nn.Linear(protein_feats, 512),  # 降维到256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),  # 降维到256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, node_out_feats)  # 进一步对齐分子图特征维度
        )

        #多头注意力预测
        self.multihead = MultiHeadAttentionClassifier(input_dim=64).to(device)

    def unbatch_node_feats(self,node_feats, batched_graph, max_nodes):

        batch_size = batched_graph.batch_size
        output_feats = []
        num_nodes = batched_graph.batch_num_nodes(ntype='_N')       #tensor（L1，L2,L3....）
        num_node_list = num_nodes.tolist()  # 将 tensor 转为列表后遍历

        # 获取每个图的节点数
        start_idx = 0
        for i in range(batch_size):
            num_node=num_node_list[i]
            # 获取该图的节点特征
            graph_node_feats = node_feats[start_idx:start_idx +num_node ]

            # 填充：如果该图的节点数少于Lmax，则用0填充
            padded_feats = torch.zeros((max_nodes, graph_node_feats.shape[1]), dtype=torch.float32)
            padded_feats[:num_node] = graph_node_feats  # 将图的节点特征填充到矩阵的前部分

            output_feats.append(padded_feats)

            # 更新当前节点的索引
            start_idx += num_node

        # 将所有图的节点特征堆叠到一起，得到 (batch_size, Lmax, C)
        return torch.stack(output_feats)



    def forward(self, graph_feats, node_feats, edge_feats, protein_feats, Ad):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """联合化合物和蛋白质特征进行预测"""
        # 处理化合物图的节点和边特征
        node_feats = self.gnn(graph_feats, node_feats, edge_feats)        #(sigL,64)

        num_nodes = graph_feats.batch_num_nodes(ntype='_N')
        max_nodes = num_nodes.max().item()
        node_feats = self.unbatch_node_feats(node_feats, graph_feats, max_nodes)        #（batchsize，Lmax，C）（64）
        node_feats = node_feats.to(device)

        # 蛋白质分支
        protein_feats = self.protein_projector(protein_feats)  # 输出形状: (batch_size, 64)

        # 蛋白-化合物特征融合

        combined_feats = self.fusion(node_feats,Ad, protein_feats)



        # 预测活性
        predict= self.multihead(combined_feats)
        return predict

