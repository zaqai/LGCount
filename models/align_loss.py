import torch
import torch.nn as nn


class AlignLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(AlignLoss, self).__init__()
        self.temperature = temperature

    def forward(self, coarse_sim_map, fine_sim_map, coarse_GT, fine_GT):
        # contrast loss
        _, pred_coarse_indices = torch.topk(coarse_sim_map, k=1, dim=1)
        pred_coarse_indices.squeeze_(-1)
        coarse_sim_map = torch.exp(coarse_sim_map / self.temperature)
        fine_sim_map = torch.exp(fine_sim_map / self.temperature)
        coarse_pos_sum = coarse_sim_map[torch.arange(coarse_sim_map.shape[0]), coarse_GT]
        coarse_sum = torch.sum(coarse_sim_map, dim=1) + 1e-5
        fine_pos_sum = fine_sim_map[torch.arange(fine_sim_map.shape[0]), fine_GT]
        fine_sum = torch.sum(fine_sim_map, dim=1) + 1e-5
        coarse_rank_loss = -torch.log(coarse_pos_sum / coarse_sum)
        fine_rank_loss = -torch.log(fine_pos_sum / fine_sum)
        # 只有当coarse预测对，fine才有意义
        fine_rank_loss[pred_coarse_indices != coarse_GT] = 2.0

        # rank_loss=coarse_rank_loss+fine_rank_loss
        # rank_loss=rank_loss.mean()
        return coarse_rank_loss.mean(), fine_rank_loss.mean()
