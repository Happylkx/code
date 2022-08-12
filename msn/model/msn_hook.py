# Copyright (c) OpenMMLab. All rights reserved.
import math
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MSNHook(Hook):
    """Hook for MSN.
    This hook is for MSN to achieve momentum(for target encoder) and 
    sharpen temperature scheduler .
    Args:

    """

    def __init__(self,
                 start_sharpen=0.25,
                 final_sharpen=0.25,
                 start_momentum=0.996,
                 final_momentum=1.0,
                 start_weight_decay=0.04,
                 final_weight_decay=0.4,
                 **kwargs):
        self._init = False

        self.start_sharpen = start_sharpen
        self.final_sharpen = final_sharpen
        self.start_momentum = start_momentum
        self.final_momentum = final_momentum
        self.start_wd = start_weight_decay
        self.final_wd = final_weight_decay

        # --------running status--------
        self._max_iters = None

    def before_train_epoch(self, runner):
        if not self._init:
            ipe = len(runner.data_loader)
            self._max_iters = ipe * runner.max_epochs
            self._init = True
            

    def before_train_iter(self, runner):
        assert self._init, "MSNHook scheduler not initialized."
        model = runner.model.module  # runner.model->MMDataParallel
        assert 'MSN' in str(type(model))

        model.T = self.get_sharpen_temperature(runner.iter)
        model.momentum = self.get_momentum(runner.iter)
        self._step_weight_decay(runner, self.get_weight_decay(runner.iter))

    def get_momentum(self, iter):
        increment = (self.final_momentum - self.start_momentum) / (self._max_iters * 1.25)
        return self.start_momentum + increment * iter

    def get_sharpen_temperature(self, iter):
        increment = (self.final_sharpen - self.start_sharpen) / (self._max_iters * 1.25)
        return self.start_sharpen + increment * iter

    def get_weight_decay(self, iter):
        progress = iter / self._max_iters
        return self.final_wd + (self.start_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

    def _step_weight_decay(self, runner, new_wd):
        if self.final_wd <= self.start_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in runner.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
