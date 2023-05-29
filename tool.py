import torch
import torch.nn.functional as F
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin  # Margin for triplet loss

    def forward(self, anchor, positive, negative):
        """
        anchor: the representation for anchor samples, shape [N, D]
        positive: the representation for positive samples, shape [N, D]
        negative: the representation for negative samples, shape [N, D]
        N is the batch size, D is the dimension of the representation
        """
        pos_dist = F.pairwise_distance(anchor, positive)  # Compute the distance between anchor and positive
        neg_dist = F.pairwise_distance(anchor, negative)  # Compute the distance between anchor and negative

        # Implement the triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)  # Apply ReLU to ensure the loss is non-negative
        return losses.mean()  # Return the mean loss


class InforNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InforNCELoss, self).__init__()
        self.temperature = temperature  # Temperature parameter for the softmax function

    def forward(self, q, k):
        """
        q: query representations, shape [N, D]
        k: key representations, shape [N, D]
        N is the batch size, D is the dimension of the representation
        """
        # Normalize the query and key representations
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # Compute the dot products for query-key pairs
        logits = torch.mm(q, k.t()) / self.temperature

        # Compute the probabilities with softmax
        probs = torch.softmax(logits, dim=-1)

        # Compute the InfoNCE loss
        labels = torch.arange(q.size(0)).to(q.device)
        loss = F.cross_entropy(logits, labels)

        return loss
