
import numpy as np
from scipy.sparse import csr_matrix


PARAMS = [
    "fast_lr", "slow_lr", "fast_decay", "slow_decay",
    "state_leak", "recurrence", "context_gate",
    "mix_bias", "mix_gain", "dopamine",
    "eligibility", "exploration"
]

LO = np.array([
    .02, .005, 0, 0, .1, 0, 0, -3, 0, 0, .4, .03
])

HI = np.array([
    1, .6, .12, .02, .99, 1.5, 8, 3, 6, 1, .999, .5
])

DEFAULT = np.array([
    .4, .12, .02, .001, .8, .8, 2, 0, 1, .5, .95, .2
])


class Circuit:
    def __init__(self, km, rec, dm):
        self.g = np.sqrt(km / km[km > 0].mean()).clip(0, 4)
        self.g = self.g.astype(np.float32)
        self.mass = self.g.sum(0).clip(min=1)

        self.rec = csr_matrix(rec.T)
        self.dm = dm
        self.m = km.shape[1]
        self.n = len(rec)

        rng = np.random.default_rng(99)
        groups = rng.permutation(self.m) % 4

        self.out = np.eye(4, dtype=np.float32)[groups]
        self.out /= np.sqrt(self.out.sum(0))


class Brain:
    def __init__(self, circuit, theta, mode="full"):
        self.c = circuit
        self.t = theta
        self.mode = mode

        self.fast = np.zeros_like(circuit.g)
        self.slow = np.zeros_like(circuit.g)
        self.h = np.zeros(circuit.n, dtype=np.float32)

    def reset(self):
        self.h.fill(0)

    def pulse(self, ids=None, strength=1):
        c, t = self.c, self.t
        z = t[4] * self.h

        if self.mode != "no feedback":
            z += t[5] * (c.rec @ self.h)

        if ids is not None:
            z[:c.m] += strength * 8 * c.g[ids].sum(0) / c.mass

        self.h = np.tanh(z).astype(np.float32)

    def decide(self, ids):
        c, t = self.c, self.t

        if self.mode == "no state":
            self.h.fill(0)

        context = self.h[:c.m]
        context = context - context.mean()

        gate = np.exp(np.clip(t[6] * context, -2, 2))
        mod = self.h[c.m:] @ c.dm

        mix = 1 / (1 + np.exp(-np.clip(
            t[7] + t[8] * mod, -10, 10
        )))

        if self.mode == "single":
            mix = np.zeros_like(mix)

        g = c.g[ids] * gate

        w = (
            (1 - mix) * self.fast[ids] +
            mix * self.slow[ids]
        )

        q = (g * w).sum(0) @ c.out
        self.pulse(ids)

        return q, (ids, g, mix, q)

    def learn(self, record, action, reward):
        if self.mode == "frozen":
            return

        c, t = self.c, self.t
        ids, g, mix, q = record

        error = np.clip(reward - q[action], -2, 2)

        mod = np.tanh(self.h[c.m:] + error) @ c.dm
        mod = 1 + t[9] * np.tanh(mod)

        if self.mode == "no dopamine":
            mod = np.ones(c.m)

        elig = t[10] ** 2
        grad = g * c.out[:, action]

        gf = grad * (1 - mix)
        gs = grad * mix

        norm = (gf**2 + gs**2).sum() + 1e-8

        df = elig * error * gf * mod / norm
        ds = elig * error * gs * mod / norm

        self.fast[ids] = np.clip(
            (1 - t[2]) * self.fast[ids] + t[0] * df,
            -3, 3
        )

        if self.mode != "single":
            self.slow[ids] = np.clip(
                (1 - t[3]) * self.slow[ids] + t[1] * ds,
                -3, 3
            )
