from collections import OrderedDict
from typing import Tuple
import torch.nn as nn
from torch.nn.init import trunc_normal_
from mmcv.runner import BaseModule

from mmselfsup.models.builder import NECKS


@NECKS.register_module()
class MSNNeck(BaseModule):
    """The MSN neck: fc only.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                hidden_dim=2048,
                output_dim=128,
                emb_dim=192,
                use_bn=True,
                init_cfg=None):
        super(MSNNeck, self).__init__(init_cfg)

        fc=OrderedDict()
        fc['fc1'] = nn.Linear(emb_dim, hidden_dim)
        if use_bn:
            fc['bn1'] = nn.BatchNorm1d(hidden_dim)
        fc['gelu1'] = nn.GELU()
        fc['fc2'] = nn.Linear(hidden_dim, hidden_dim)
        if use_bn:
            fc['bn2'] = nn.BatchNorm1d(hidden_dim)
        fc['gelu2'] = nn.GELU()
        fc['fc3'] = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Sequential(fc)

        # init
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        return self.fc(x)
