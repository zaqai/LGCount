import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(RankLoss, self).__init__()
        self.temperature = temperature

    def forward(self, patch_embedding, class_text_embedding, gt_density):
        """
        Args:
            patch_embedding: (B, 196, 512) embedding of image patch feature
            text_embedding: (B, 1, 512), ground truth text embedding
            gt_density: (B, 384, 384), ground truth density map
        """

        gt_density = F.interpolate(gt_density.unsqueeze_(1), size=(224, 224), mode='nearest')
        density_mask = F.max_pool2d(gt_density, kernel_size=16, stride=16, padding=0)  # same as ViT conv1
        density_mask = density_mask > 0.
        density_mask = density_mask.permute(0, 2, 3, 1)  # (B, 14, 14, 1)

        class_text_embedding = class_text_embedding.unsqueeze(1).expand(-1, 14, 14, -1)

        # [B, 14, 14, 512], contains both gt and noise text embedding
        fused_text_embedding_map = class_text_embedding
        pos_mask = density_mask.squeeze_(-1)  # (B, 14, 14, 1)

        patch_embeddings = patch_embedding.reshape(-1, 14, 14, 512)
        sim_map = F.cosine_similarity(patch_embeddings, fused_text_embedding_map, dim=-1)  # (B, 14, 14)

        sim_map = torch.exp(sim_map / self.temperature)

        rank_loss_list = []
        for i in range(sim_map.shape[0]):
            positive_scores = sim_map[i][pos_mask[i]]
            negative_scores = sim_map[i][~pos_mask[i]]
            # 没有前景块或背景块
            if positive_scores.shape[0] == 0 or negative_scores.shape[0] == 0:
                continue
            pos_neg_diff = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
            rank_loss = torch.clamp(2.0 - pos_neg_diff, min=0).mean()
            rank_loss_list.append(rank_loss)

        rank_loss_stack = torch.stack(rank_loss_list)
        loss = rank_loss_stack.mean()

        return loss
