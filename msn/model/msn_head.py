"""
mmselfsup/models/heads/__init__.py

from .customized_head import CustomizedHead

__all__ = [..., CustomizedHead, ...]
"""
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import BaseModule, get_dist_info
#from mmselfsup.utils.distributed_sinkhorn import distributed_sinkhorn

from mmselfsup.models.builder import HEADS


@HEADS.register_module()
class MSNHead(BaseModule):
    """
    https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/losses.py
    Make unsupervised MSN loss
    :num_views: number of anchor views
    :param tau: cosine similarity temperature
    :param me_max: whether to perform me-max regularization
    :param return_preds: whether to return anchor predictions
    """

    def __init__(self,
                 num_views=1,
                 tau=0.1,
                 me_max=True,
                 me_max_weight=1,
                 use_entropy=True,
                 ent_weight=1,
                 return_preds=False,
                 use_sinkhorn=True,
                 ):
        super(MSNHead, self).__init__()
        # --------Hyper Parameters-------
        self.num_views = num_views
        self.tau = tau
        self.me_max = me_max  # Switch
        self.me_max_weight = me_max_weight
        self.ent_weight = ent_weight
        self.return_preds = return_preds
        self.use_sinkhorn = use_sinkhorn
        self.use_entropy = use_entropy
        # --------Model Structure--------
        self.softmax = torch.nn.Softmax(dim=1)

    """def forward(self, anchor_views, target_views,):
        # Nothing needs to be done here, since we only need the loss.
        return anchor_views, target_views"""

    def forward(self,
                anchor_views,
                target_views,
                prototypes,
                proto_labels,
                T=0.25,

                ):
        """Compute the loss."""
        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, prototypes, proto_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.sharpen(self.snn(target_views, prototypes, proto_labels), T=T)
            if self.use_sinkhorn:
                # rank, world_size = get_dist_info()
                # targets = distributed_sinkhorn(targets,sinkhorn_iterations=3,world_size=world_size,epsilon=1)
                targets = distributed_sinkhorn(targets)
            targets = torch.cat([targets for _ in range(self.num_views)], dim=0)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if self.me_max:
            avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
            rloss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))

        sloss = 0.
        if self.use_entropy:
            sloss = torch.mean(torch.sum(torch.log(probs ** (-probs)), dim=1))

        # -- logging
        with torch.no_grad():
            num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            self.log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

        if self.return_preds:
            return loss, rloss, sloss, self.log_dct, targets

        losses = dict()
        losses['ce_loss'] = loss
        losses['memax_loss'] = self.me_max_weight * rloss
        losses['ent_loss'] = self.ent_weight * sloss

        return losses

    def sharpen(self, p, T):
        sharp_p = p ** (1. / T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(self, query, supports, support_labels):
        """ Soft Nearest Neighbours similarity classifier """
        temp = self.tau
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        return self.softmax(query @ supports.T / temp) @ support_labels


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        """if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):"""
        rank, world_size = get_dist_info()
        if world_size > 1:
            x = x.contiguous() / dist.get_world_size()  # all_reduce sums up the tensors, so divide first. op=dist.ReduceOp.SUM by default
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


# https://github.com/open-mmlab/mmselfsup/blob/47f6feb9251a7e9912b65deb1a46779922c403e7/mmselfsup/utils/distributed_sinkhorn.py

@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    rank, world_size = get_dist_info()
    _got_dist = use_dist and world_size > 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T
