
import torch
import torch.nn.functional as F


class Memory:
    def __init__(self, w, lr=0.15):
        self.scale = (w / w[w > 0].mean()).clamp(max=4)
        self.delta = torch.zeros_like(w)
        self.lr = lr

    def logits(self, x):
        return x @ (self.scale * self.delta)

    def predict(self, x):
        return self.logits(x).argmax(-1)

    def learn(self, x, y, freeze=None):
        p = self.logits(x).softmax(-1)
        target = F.one_hot(torch.tensor(y), 2).float()
        error = target - p
        update = self.lr * x[:, None] * self.scale * error / x.sum()

        if freeze is not None:
            update[freeze] = 0

        self.delta += update
