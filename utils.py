import torch
import torch.nn.functional as F
from torch import nn

class Accuracy:
    def __init__(self, model, num_train=1, num_val=1, topk=None):
        self.model = model
        self.topk = topk
        self.num_train = num_train
        self.num_val = num_val
        self.corrects_train = 0
        self.corrects_train_k = 0
        self.corrects_val = 0
        self.corrects_val_k = 0

    def update(self, y_pred, y):
        if self.model.training:
            self.corrects_train += (y_pred.argmax(-1)==y).sum().item()
            if self.topk:
                topk = y_pred.topk(self.topk)[1]
                self.corrects_train_k += y.view(-1,1).expand_as(topk).eq(topk).sum().item()
        else:
            self.corrects_val += (y_pred.argmax(-1)==y).sum().item()
            if self.topk:
                topk = y_pred.topk(self.topk)[1]
                self.corrects_val_k += y.view(-1,1).expand_as(topk).eq(topk).sum().item()

    def __repr__(self):
        top1 = self.corrects_train/self.num_train*100, self.corrects_val/self.num_val*100
        if not self.topk:
            return "{:.2f} {:.2f}".format(*top1)
        else:
            topk = self.corrects_train_k/self.num_train*100, self.corrects_val_k/self.num_val*100
            return "(top1) {:.2f} {:.2f} | (top{}) {:.2f} {:.2f}".format(*top1, self.topk, *topk)


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
