
class Memory:
    def __init__(self, w):
        target = int(w.sum(0).argmax())
        self.w = w[:, target].clone()
        self.w /= self.w.sum().clamp_min(1)

    def respond(self, odor):
        return float(odor @ self.w)

    def learn(self, odor, reward=1):
        self.w[odor.bool()] *= 1 + 0.1 * reward
