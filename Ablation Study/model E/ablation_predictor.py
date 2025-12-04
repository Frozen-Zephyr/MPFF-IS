from encoders import *
import torch
import torch.nn.functional as F


class MLPMixer(nn.Module):
    def __init__(self, input_dim, num_patches=4, token_dim=128, channel_dim=128, n_classes=1):
        super().__init__()
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
        x = x.view(B, self.num_patches, self.patch_dim)  # [B, N, C]

        # token mixing: transpose [B, N, C] -> [B, C, N]
        y = x.transpose(1, 2)
        y = self.token_mixers(y)
        x = x + y.transpose(1, 2)  # residual

        # channel mixing
        y = self.channel_mixers(x)
        x = x + y  # residual

        x = x.flatten(1)  # [B, N*C] = [B, D]
        return self.classifier(x)  # 回归任务时输出 [B, 1]

class PredictorWithProtein(nn.Module):
    def __init__(self,pro_dim=20, lig_dim=128, d_model=128, nhead=4, num_classes=1,
                 node_out_feats=64,
                 n_tasks=1,):
        super().__init__()
        input_dim = pro_dim + lig_dim

        # 预测层
        self.predict = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 256),  # 进一步增加深度
            nn.ReLU(),
            nn.Linear(256, node_out_feats),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, pro_feats,lig_feats):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """联合化合物和蛋白质特征进行预测"""

        # 对齐序列长度
        max_len = max(pro_feats.shape[1], lig_feats.shape[1])
        pro_padded = F.pad(pro_feats, (0, 0, 0, max_len - pro_feats.shape[1]))
        lig_padded = F.pad(lig_feats, (0, 0, 0, max_len - lig_feats.shape[1]))
        # 拼接特征
        combined_feats = torch.cat([pro_padded, lig_padded], dim=-1)  # (batch, max_len, d1+d2)
        # 预测活性
        predict = self.predict(combined_feats)
        # predict = predict.view(-1, 1)
        predict = torch.mean(predict, dim=1)


        return predict

