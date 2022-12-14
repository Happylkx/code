import copy
from itertools import chain
from mmcv.runner import get_dist_info
from typing import Optional, Tuple
import torch
import torch.nn as nn

from mmselfsup.models.builder import ALGORITHMS, build_backbone, build_head, build_neck
from mmselfsup.models.algorithms.base import BaseModel


@ALGORITHMS.register_module()
class MSN(BaseModel):
    """Masked Siamese Network.

    Implementation of `Masked Siamese Network
    <https://arxiv.org/abs/2204.07141>`_.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for neck. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,

                 output_dim: int,
                 patch_drop: float,
                 num_proto: int = 2048,

                 freeze_proto: bool = False,
                 sync_batchnorm: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super(MSN, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        # --------Hyper Parameters--------
        self.patch_drop = patch_drop
        self.output_dim = output_dim
        self.num_proto = num_proto
        self.freeze_proto = freeze_proto
        self.sync_batchnorm = sync_batchnorm
        # --------Scheduled Values, injected by MSNHook--------
        self.momentum = None
        self.T = None  # Sharpen
        # --------init--------
        self.prototypes, self.proto_labels = self.build_prototypes(num_proto, output_dim, freeze_proto)
        self.encoder = self.backbone
        del self.backbone  # Avoid false alarm when loading checkpoint
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_neck = copy.deepcopy(self.neck)
        for p in chain(self.target_encoder.parameters(), self.target_neck.parameters()):
            p.requires_grad = False
        if self.sync_batchnorm:
            rank, world_size = get_dist_info()
            if world_size > 1:
                self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.target_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.target_encoder)
                self.neck = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.neck)
                self.target_neck = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.target_neck)

    def build_prototypes(self, num_proto, output_dim, freeze_proto):
        # -- make prototypes
        prototypes, proto_labels = None, None
        if num_proto > 0:
            with torch.no_grad():
                prototypes = torch.empty(num_proto, output_dim)
                _sqrt_k = (1. / output_dim) ** 0.5
                torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
                prototypes = torch.nn.parameter.Parameter(
                    prototypes)  # Convert this to Parameter, otherwise it cannot be recognized by frameworks for saving, etc.

                # -- init prototype labels
                proto_labels = one_hot(torch.tensor([i for i in range(num_proto)]), num_proto)

            if not freeze_proto:
                prototypes.requires_grad = True
        return prototypes, proto_labels

    def extract_feat(self, imgs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.
        Args:
            imgs (list): Input images of shape (N, C, H, W).
        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        # return self.backbone(imgs, patch_drop=self.patch_drop)
        raise NotImplementedError()

    def forward_multiple_sizes(self, x, encoder, patch_drop=0.) -> list:
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        results = []
        start_idx = 0
        for end_idx in idx_crops:
            h = encoder(torch.cat(x[start_idx:end_idx]), patch_drop)
            results.append(h)
            # only rand views need patch_drop
            patch_drop = 0.
            start_idx = end_idx

        return results

    def _forward_neck(self, x: list, neck):
        # x:[(repr,), (repr,), ...], each tuple is a repr list of rand/focal view
        assert isinstance(x, list) and isinstance(x[0], tuple)
        z = None
        for reps in x:
            batch = self._get_repr_from_tuple(reps)
            _z = neck(batch)
            if z is not None:
                z = torch.cat((z, _z))
            else:
                z = _z
        return z

    def forward_train(self, imgs: list, **kwargs):
        # Move proto_labels to according device
        if not (self.proto_labels.device == imgs[0].device):
            self.proto_labels = self.proto_labels.to(imgs[0].device)

        # Step 0(5). momentum update of target encoder
        self._momentum_update(self.encoder, self.target_encoder)
        self._momentum_update(self.neck, self.target_neck)

        # Get representations of target view h, and anchor view z
        # h: representations of 'imgs' before head
        # z: representations of 'imgs' after head

        # Encoder returns a tuple, (representations, )
        reprs = self.forward_multiple_sizes(imgs[1:], self.encoder, patch_drop=self.patch_drop)
        z = self._forward_neck(reprs, self.neck)
        with torch.no_grad():
            # target views do not need drop_patch
            a = self.target_encoder(imgs[0])
            h = self.target_neck(self._get_repr_from_tuple(a))

        # Step 1. convert representations to fp32
        h, z = h.float(), z.float()

        # Step 2. determine anchor views/supports and their
        #         corresponding target views/supports
        # --
        anchor_views, target_views = z, h.detach()

        # Step 3. compute msn loss with me-max regularization
        losses = self.head(T=self.T,
                           anchor_views=anchor_views,
                           target_views=target_views,
                           proto_labels=self.proto_labels,
                           prototypes=self.prototypes)

        # Schedules are handled in MSNHook

        return losses

    def _get_repr_from_tuple(self, x):
        assert isinstance(x, (tuple, list))
        return x[0]

    def forward_test(self, x):
        raise NotImplementedError()

    # ??????https://github.com/open-mmlab/mmselfsup/blob/master/mmselfsup/models/algorithms/moco.py#L68
    @torch.no_grad()
    def _momentum_update(self, src, target):
        m = self.momentum
        for param_q, param_k in zip(src.parameters(), target.parameters()):
            param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)


def one_hot(targets, num_classes, smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    targets = targets.long().view(-1, 1)  # .to(device)
    return torch.full((len(targets), num_classes), off_value,  # device=device
                      ).scatter_(1, targets, on_value)
