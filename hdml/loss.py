import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, size_average=True):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean() if size_average else losses.sum()


def cross_entropy(logits, label, size_average=True):
    if size_average:
        return torch.mean(torch.sum(-label * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(-label * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, x, label):
        anchor, positive = torch.split(x, 2, dim=0)
        batch_size = anchor.size(0)
        label = label.view(label.size(0), 1)

        label = (label == torch.transpose(label, 0, 1)).float()
        label = label / torch.sum(label, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, label)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss
