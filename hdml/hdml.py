import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import googlenet
from . import loss


class Generator(nn.Module):
    def __init__(self, in_channel=128, out_channel=1024):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_channel, out_channel//2)
        self.bn1 = nn.BatchNorm1d(out_channel//2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_channel//2, out_channel)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        return out


class NPairBase(nn.Module):
    def __init__(self, embedding_size=128, n_class=99, pretrained=False):
        super(NPairBase, self).__init__()
        n_mid = 1024
        self.googlenet = googlenet.googlenet(pretrained=pretrained)
        self.bn1 = nn.BatchNorm1d(n_mid)
        self.fc1 = nn.Linear(n_mid, embedding_size)
        self.loss_fn = loss.NpairLoss()

    def forward(self, x, label):
        embedding_y_orig = self.googlenet(x)
        embedding = self.bn1(embedding_y_orig)
        embedding_z = self.fc1(embedding)
        jm = self.loss_fn(embedding_z, label)
        return jm, embedding_y_orig, embedding_z


class TripletBase(nn.Module):
    def __init__(self, embedding_size=128, n_class=99, pretrained=False):
        super(TripletBase, self).__init__()
        n_mid = 1024
        self.googlenet = googlenet.googlenet(pretrained=pretrained)
        self.bn1 = nn.BatchNorm1d(n_mid)
        self.fc1 = nn.Linear(n_mid, embedding_size)
        self.loss_fn = loss.TripletLoss()

    def forward(self, x, use_loss=True):
        embedding_y_orig = self.googlenet(x)
        embedding = self.bn1(embedding_y_orig)
        embedding_z = self.fc1(embedding)
        if use_loss:
            jm = self.loss_fn(embedding_z)
            return jm, embedding_y_orig, embedding_z
        return embedding_z


class TripletPulling(nn.Module):
    def __init__(self, embedding_size=128, alpha=90.0):
        super(TripletPulling, self).__init__()
        self.embedding_size = embedding_size
        self.alpha = alpha

    def forward(self, x, jm):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)
        r = (dist_pos + (dist_neg - dist_pos) * np.exp(-self.alpha / jm) / dist_neg).unsqueeze(-1).repeat(1, self.embedding_size)
        neg2 = anchor + torch.mul((negative - anchor), r)
        neg_mask = torch.ge(dist_pos, dist_neg)
        op_neg_mask = ~neg_mask
        neg_mask = neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        op_neg_mask = op_neg_mask.float().unsqueeze(-1).repeat(1, self.embedding_size)
        neg_hat = torch.mul(negative, neg_mask) + torch.mul(neg2, op_neg_mask)
        return torch.cat([anchor, positive, neg_hat], dim=0)


class TripletHDML(nn.Module):
    def __init__(self, embedding_size=128, n_class=99,
                 beta=1e+4, lmd=0.5, softmax_factor=1e+4,
                 pretrained=False):
        super(TripletHDML, self).__init__()
        n_mid = 1024
        self.beta = beta
        self.lmd = lmd
        self.softmax_factor = softmax_factor
        self.classifier1 = TripletBase(embedding_size, n_class, pretrained)
        self.loss_fn = loss.TripletLoss()
        self.pulling = TripletPulling()
        self.generator = Generator()
        self.classifier2 = nn.Sequential(nn.BatchNorm1d(n_mid),
                                         nn.Linear(n_mid, embedding_size))
        self.softmax_classifier = nn.Linear(n_mid, n_class)

    def forward(self, x, t, javg, jgen):
        jm, embedding_y_orig, embedding_z = self.classifier1(x)

        embedding_z_quta = self.pulling(embedding_z, javg)

        embedding_z_concate = torch.cat([embedding_z, embedding_z_quta], dim=0)
        embedding_z_concate = self.generator(embedding_z_concate)
        embedding_yp, embedding_yq = torch.chunk(embedding_z_concate, 2, dim=0)

        embedding_z_quta = self.classifier2(embedding_yq)

        e_bj = np.exp(-self.beta / jgen)
        jsyn = (1.0 - e_bj) * self.loss_fn(embedding_z_quta)
        jm = e_bj * jm
        jmetric = jm + jsyn
        logits_orig = self.softmax_classifier(embedding_y_orig)
        label = t.squeeze(-1)
        ce = F.nll_loss(F.log_softmax(logits_orig, dim=1), label, reduction='mean')

        jrecon = (1.0 - self.lmd) * (embedding_yp - embedding_y_orig).pow(2).sum()
        logits_q = self.softmax_classifier(embedding_yq)
        jsoft = self.softmax_factor * self.lmd * F.nll_loss(F.log_softmax(logits_q, dim=1), label, reduction='mean')
        jgen = jrecon + jsoft

        return jgen, jmetric, jm, ce, embedding_z