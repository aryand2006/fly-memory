
import torch


class Memory:
    def __init__(self, w, lr=0.2):
        i = w.sum(0).topk(2).indices
        a, b = w[:, i].T

        self.base = a / a.sum().clamp_min(1)
        self.s0 = b / b.sum().clamp_min(1)
        self.active = self.base.clone()
        self.silent = self.s0.clone()
        self.lr = lr

    def learn(self, x):
        k = x.bool()
        self.active[k] *= 1 - self.lr
        self.silent[k] *= 1 - self.lr

    def forget(self):
        self.active = self.base.clone()

    def remind(self, cue):
        trace = torch.where(
            self.s0 > 0, 1 - self.silent / self.s0.clamp_min(1e-12), 0
        ).clamp(0, 1)
        self.active *= 1 - cue * trace

    def score(self, x):
        return (x @ (self.base - self.active)) / (x @ self.base).clamp_min(1e-12)

    def lesion(self, seed=None):
        change = (self.s0 - self.silent).clamp_min(0)
        ids = change.nonzero().flatten()
        n = max(1, len(ids) // 4)

        if seed is None:
            cut = change.topk(n).indices
        else:
            g = torch.Generator().manual_seed(seed)
            cut = ids[torch.randperm(len(ids), generator=g)[:n]]

        self.silent[cut] = self.s0[cut]
