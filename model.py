
import torch


class Memory:
    def __init__(self, w):
        idx = w.sum(0).topk(2).indices
        self.base = w[:, idx].clone()
        self.base /= self.base.sum(0).clamp_min(1)

        self.fast = torch.zeros_like(self.base)
        self.slow = torch.zeros_like(self.base)

    def learn(self, x, y):
        k = x.bool()
        sign = torch.tensor([1., -1.] if y == 0 else [-1., 1.])
        target = self.base[k] * sign

        self.fast[k] = 0.4 * self.fast[k] + 0.6 * target
        self.slow[k] = 0.96 * self.slow[k] + 0.04 * target

    def predict(self, x, cue=None, mode="slow"):
        scores = x @ self.fast

        if cue is not None:
            w = self.slow if mode == "slow" else self.fast
            scores += 2 * (cue @ w)

        return scores.argmax(-1)

    def lesion(self, seed=None):
        w = self.slow.flatten()
        ids = w.abs().nonzero().flatten()
        n = max(1, len(ids) // 4)

        if seed is None:
            cut = ids[w[ids].abs().topk(n).indices]
        else:
            g = torch.Generator().manual_seed(seed)
            cut = ids[torch.randperm(len(ids), generator=g)[:n]]

        w[cut] = 0
