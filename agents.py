
import numpy as np
from brain import Circuit


class Linear:
    def __init__(self, n, lr):
        self.w = np.zeros((2, n, 4), dtype=np.float32)
        self.lr = lr
        self.reset()

    def reset(self):
        self.cue = np.array([], dtype=int)

    def pulse(self, ids=None, strength=1):
        if ids is not None and strength == 1:
            self.cue = ids

    def decide(self, ids):
        q = (
            self.w[0, self.cue].sum(0) +
            self.w[1, ids].sum(0)
        )
        return q, (self.cue, ids, q)

    def learn(self, record, action, reward):
        cue, ids, q = record
        change = (
            self.lr * (reward - q[action]) /
            max(1, len(cue) + len(ids))
        )

        self.w[0, cue, action] += change
        self.w[1, ids, action] += change


class Associative:
    def __init__(self, lr):
        self.lr = lr
        self.memory = {}
        self.reset()

    def reset(self):
        self.cue = np.array([], dtype=np.int32)

    def pulse(self, ids=None, strength=1):
        if ids is not None and strength == 1:
            self.cue = ids

    def decide(self, ids):
        key = (self.cue.tobytes(), ids.tobytes())

        if key in self.memory:
            q = self.memory[key][2].copy()

        elif self.memory:
            def similarity(item):
                c, k, _ = item

                a = np.intersect1d(
                    c, self.cue, assume_unique=True
                ).size / max(1, len(c), len(self.cue))

                b = np.intersect1d(
                    k, ids, assume_unique=True
                ).size / max(1, len(k), len(ids))

                return (1 + a) * b

            q = max(
                self.memory.values(), key=similarity
            )[2].copy()

        else:
            q = np.zeros(4)

        return q, (key, self.cue.copy(), ids.copy())

    def learn(self, record, action, reward):
        key, cue, ids = record

        if key not in self.memory:
            self.memory[key] = (cue, ids, np.zeros(4))

        q = self.memory[key][2]
        q[action] += self.lr * (reward - q[action])


def architectures(km, rec, dm, code):
    real = Circuit(km, rec, dm)
    freq = code.frequency(len(km))

    _, groups = np.unique(
        freq, axis=0, return_inverse=True
    )

    groups = groups.ravel()

    matched, rewired, changes = [], [], []

    for seed in (71, 72):
        rng = np.random.default_rng(seed)
        perm = np.arange(len(km))

        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            perm[ids] = rng.permutation(ids)

        w = km[perm]
        matched.append(Circuit(w, rec, dm))

        used = (km > 0).any(1)
        changes.append((w[used] != km[used]).any(1).mean())

        m = km.shape[1]

        order = np.concatenate([
            rng.permutation(m),
            np.arange(m, len(rec))
        ])

        rewired.append(Circuit(
            km[rng.permutation(len(km))],
            rec[np.ix_(order, order)],
            dm[:, order[:m]]
        ))

    print(
        "activity-matched rows changed:",
        f"{np.mean(changes):.1%}"
    )

    return {
        "real": [real],
        "matched": matched,
        "rewired": rewired,
        "single": [real]
    }
