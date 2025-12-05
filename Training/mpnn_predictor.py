import dgl
import torch
import torch.nn as nn
from mpnn import MPNNGNN
from Feature_Fusion import Fusion

class MLPMixer(nn.Module):
    def __init__(self, input_dim, num_patches=4, token_dim=128, channel_dim=128, n_classes=1):
        super(MLPMixer, self).__init__()
        assert input_dim % num_patches == 0, "input_dim must be divisible by num_patches"
        self.num_patches = num_patches
        self.patch_dim = input_dim // num_patches

        self.token_mixers = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches)
        )

        self.channel_mixers = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, self.patch_dim)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, n_classes)
        )

    def forward(self, x):
        # x: [B, D]
        B, D = x.shape
        x = x.view(B, self.num_patches, self.patch_dim)

        # token mixing: transpose [B, N, C] -> [B, C, N]
        y = x.transpose(1, 2)
        y = self.token_mixers(y)
        x = x + y.transpose(1, 2)

        # channel mixing
        y = self.channel_mixers(x)
        x = x + y

        x = x.flatten(1)  # [B, N*C] = [B, D]
        return self.classifier(x)

class MPNNPredictorWithProtein(nn.Module):
    def __init__(self,
                 node_in_feats=74,
                 edge_in_feats=12,
                 protein_feats=1280,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super().__init__()
        self.fusion = Fusion().to(device)

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing,
                           )

        self.protein_projector = nn.Sequential(
            nn.Linear(protein_feats, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, node_out_feats)
        )

        self.predict = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, node_out_feats),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(node_out_feats, n_tasks)
        )

    def unbatch_node_feats(self,node_feats, batched_graph, max_nodes):

        batch_size = batched_graph.batch_size
        output_feats = []
        num_nodes = batched_graph.batch_num_nodes(ntype='_N')
        num_node_list = num_nodes.tolist()

        # Get the number of nodes in each graph
        start_idx = 0
        for i in range(batch_size):
            num_node=num_node_list[i]
            # Obtain the node features of the graph
            graph_node_feats = node_feats[start_idx:start_idx +num_node ]

            padded_feats = torch.zeros((max_nodes, graph_node_feats.shape[1]), dtype=torch.float32)
            padded_feats[:num_node] = graph_node_feats  # Fill the front part of the matrix with the node features of the graph.
            output_feats.append(padded_feats)

            # Update the index of the current node.
            start_idx += num_node

        return torch.stack(output_feats)



    def forward(self, graph_feats, node_feats, edge_feats, protein_feats, Ad):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """Prediction based on combined compound and protein features"""
        # Processing node and edge features of compound graphs
        node_feats = self.gnn(graph_feats, node_feats, edge_feats)

        num_nodes = graph_feats.batch_num_nodes(ntype='_N')
        max_nodes = num_nodes.max().item()
        node_feats = self.unbatch_node_feats(node_feats, graph_feats, max_nodes)
        node_feats = node_feats.to(device)

        # Protein branching
        protein_feats = self.protein_projector(protein_feats)

        # Protein-compound feature fusion
        combined_feats = self.fusion(node_feats,Ad, protein_feats)

        # Predict activity
        predict= self.predict(combined_feats)
        predict=torch.mean(predict,dim=1)
        return predict

