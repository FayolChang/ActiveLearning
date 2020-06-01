import torch
import torch.nn as nn


class LabelSmoothLoss(nn.Module):
    def __init__(self, num_labels, smoothing=0.2, dim=-1):
        super(LabelSmoothLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_labels = num_labels
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)  # [b,N]
        with torch.no_grad():
            true_dist = torch.zeros_like(pred, device=pred.device)
            true_dist.fill_(self.smoothing / self.num_labels)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))




