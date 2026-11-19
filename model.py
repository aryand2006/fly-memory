
import torch
import torch.nn.functional as F


class Memory:
    def __init__(self, w, lr=0.15):
        self.scale = (w / w[w > 0].mean()).clamp(max=4)
        self.delta = torch.zeros_like(w)
        self.lr = lr

    def logits(self, x, lesion=None):
        s = x @ (self.scale * self.delta)
        if lesion is not None:
            s[..., lesion] = 0
        return s

    def predict(self, x, lesion=None):
        return self.logits(x, lesion).argmax(-1)

    def learn(self, x, y):
        p = self.logits(x).softmax(-1)
        target = F.one_hot(torch.tensor(y), 2).float()
        error = target - p
        self.delta += self.lr * x[:, None] * self.scale * error / x.sum()
