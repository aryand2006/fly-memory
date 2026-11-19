
class Memory:
    def __init__(self, w, lr=0.15):
        idx = w.sum(0).topk(2).indices
        self.w = w[:, idx].clone()
        self.w /= self.w.sum(0).clamp_min(1)
        self.lr = lr

    def predict(self, x):
        return (x @ self.w).argmin(-1)

    def learn(self, x, label):
        self.w[x.bool(), label] *= 1 - self.lr
