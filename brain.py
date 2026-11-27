import numpy as np
from scipy.sparse import csr_matrix


class Brain:
    def __init__(self, km, rec, dm, seed, mode="full"):
        self.mode = mode
        self.m = km.shape[1]

        self.gain = np.sqrt(
            km / km[km > 0].mean()
        ).clip(0, 4).astype(np.float32)

        self.fast = np.zeros_like(self.gain)
        self.slow = np.zeros_like(self.gain)

        self.rec = csr_matrix(rec.T)
        self.dm = dm
        self.mass = self.gain.sum(0).clip(min=1)

        self.h = np.zeros(len(rec), dtype=np.float32)

        rng = np.random.default_rng(seed)
        groups = rng.permutation(self.m) % 2

        self.out = np.eye(2, dtype=np.float32)[groups]
        self.out /= np.sqrt(self.out.sum(0))

    def reset(self):
        self.h.fill(0)

    def values(self, ids, key=None):
        g = self.gain[ids]
        z = ((self.fast[ids] + self.slow[ids]) * g).sum(0)

        drive = np.zeros_like(self.h)
        drive[:self.m] = 10 * g.sum(0) / self.mass + 0.5 * z

        if self.mode == "no feedback":
            self.h = np.tanh(drive)
        else:
            for _ in range(2):
                self.h = np.tanh(drive + 0.8 * (self.rec @ self.h))

        q = (z + 0.1 * self.h[:self.m]) @ self.out
        record = (ids, g, q.copy(), self.h[self.m:].copy())

        return q, record

    def learn(self, records, reward):
        for record, action in records:
            ids, g, q, dan = record

            error = np.clip(reward - q[action], -2, 2)
            grad = g * self.out[:, action]

            if self.mode == "no dopamine":
                mod = np.ones(self.m, dtype=np.float32)
            else:
                signal = np.tanh(dan + error)
                mod = 1 + 0.5 * np.tanh(signal @ self.dm)

            norm = np.sum(grad * grad * mod) + 1e-8
            change = 0.3 * error * grad * mod / norm

            f = self.fast[ids]
            s = self.slow[ids]

            if self.mode != "persistent":
                f *= 0.98

            f += change

            if self.mode not in ("no slow", "persistent"):
                transfer = 0.05 * f * mod
                f -= transfer
                s += transfer

            self.fast[ids] = f
            self.slow[ids] = s
