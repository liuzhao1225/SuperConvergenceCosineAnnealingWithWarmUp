import torch
import math
class myScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, peak_it, total_it, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.peak_it = peak_it
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.last_epoch = last_epoch
        self.total_it = total_it
        super(myScheduler, self). __init__(optimizer, last_epoch, verbose)
    def get_lr(self):
        if self.last_epoch > self.peak_it:
            return [(math.cos((self.last_epoch - self.peak_it)*math.pi/(self.total_it - self.peak_it)) + 1)/2 * self.max_lr for group in self.optimizer.param_groups]
        else:
            return [(-math.cos(self.last_epoch*math.pi/self.peak_it) + 1)/2 * (self.max_lr - self.base_lr) + self.base_lr for group in self.optimizer.param_groups]

