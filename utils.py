import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_epoch(state, save_path, epoch, checkpoint, is_best_acc, is_best_f1):
    if checkpoint:
        name = 'epoch_' + str(epoch) + '.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('Checkpoint saved:', name)

    if is_best_acc:
        name = 'epoch_best_acc.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('New best model saved')

    if is_best_f1:
        name = 'epoch_best_f1.pth.tar'
        torch.save(state, os.path.join(save_path, name))
        print('New best model saved')

    name = 'epoch_last.pth.tar'
    torch.save(state, os.path.join(save_path, name))
