"""Paper point loss plus reconstructed auxiliary supervision.

Coordinates are (x,y) pixels. Positive targets are ones; negative targets
are zeros. The public PDF's positive log-probability signs are treated as
negative log likelihood, consistent with public cross-entropy code.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class PointCriterion(nn.Module):
    def __init__(self, point_weight=.0002, aux_weight=.2, eos_coef=.5,
                 cost_class=1., cost_point=.05):
        super().__init__()
        self.point_weight = point_weight
        self.aux_weight = aux_weight
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.register_buffer('class_weight', torch.tensor([eos_coef, 1.]))

    @torch.no_grad()
    def match(self, outputs, targets):
        assignments = []
        # Match within each image rather than allocate a BQ x all-targets
        # matrix containing cross-image costs that would later be discarded.
        for logits, points, target in zip(outputs['pred_logits'], outputs['pred_points'], targets):
            truth = target['points']
            if len(truth) > len(points):
                raise ValueError('More targets than point proposals; increase proposal density')
            if not len(truth):
                empty = torch.empty(0, dtype=torch.long, device=points.device)
                assignments.append((empty, empty))
                continue
            cost = self.cost_point * torch.cdist(points.float(), truth.float())
            cost -= self.cost_class * logits.float().softmax(-1)[:, 1:2]
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            assignments.append((torch.as_tensor(rows, device=points.device),
                                torch.as_tensor(cols, device=points.device)))
        return assignments

    def forward(self, outputs, targets):
        logits = outputs['pred_logits'].float()
        points = outputs['pred_points'].float()
        if not torch.isfinite(logits).all() or not torch.isfinite(points).all():
            raise FloatingPointError('Non-finite predictions; try FP32 if AMP is enabled')
        labels = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.long)
        location = points.sum() * 0.
        matched_count = 0
        for batch, (source, destination) in enumerate(self.match(outputs, targets)):
            labels[batch, source] = 1
            location = location + F.mse_loss(points[batch, source],
                                             targets[batch]['points'][destination], reduction='sum')
            matched_count += len(destination)
        point_loss = location / max(1, matched_count)
        class_loss = F.cross_entropy(logits.transpose(1, 2), labels, self.class_weight)
        pos_losses, neg_losses = [], []
        for record in outputs.get('aux', []):
            positive = record['kind'] == 'pos'
            aux_logits = record['pred_logits'].float().reshape(-1, 2)
            aux_labels = torch.full((len(aux_logits),), int(positive),
                                    dtype=torch.long, device=logits.device)
            cls = F.cross_entropy(aux_logits, aux_labels)
            delta = ((record['pred_points'].float() - record['targets'].float())
                     if positive else record['offset'].float())
            loc = delta.square().sum(-1).mean()
            (pos_losses if positive else neg_losses).append(cls + self.point_weight * loc)
        zero = logits.sum() * 0.
        positive = torch.stack(pos_losses).mean() if pos_losses else zero
        negative = torch.stack(neg_losses).mean() if neg_losses else zero
        auxiliary = positive + negative
        total = class_loss + self.point_weight * point_loss + self.aux_weight * auxiliary
        return dict(loss=total, loss_ce=class_loss, loss_points=point_loss,
                    loss_aux=auxiliary, loss_aux_pos=positive, loss_aux_neg=negative)
