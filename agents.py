
import numpy as np


class Linear:
    def __init__(self, n):
        self.w = np.zeros((n, 2), dtype=np.float32)

    def reset(self):
        pass

    def values(self, ids, key=None):
        q = self.w[ids].sum(0)
        return q, (ids, q.copy())

    def learn(self, records, reward):
        for (ids, q), a in records:
            self.w[ids, a] += 0.3 * (reward - q[a]) / len(ids)


class Tabular:
    def __init__(self):
        self.q = np.zeros((5, 6, 3, 2), dtype=np.float32)

    def reset(self):
        pass

    def values(self, ids, key):
        q = self.q[key].copy()
        return q, (key, q)

    def learn(self, records, reward):
        for (key, q), a in records:
            self.q[key + (a,)] += 0.3 * (reward - q[a])


def controls(km, rec, dm, features, seed):
    rng = np.random.default_rng(seed)

    shuffled = km[rng.permutation(len(km))]

    freq = np.stack([
        np.bincount(
            features[c, :, :, 0].flatten(),
            minlength=len(km)
        )
        for c in range(4)
    ], axis=1)

    _, groups = np.unique(freq, axis=0, return_inverse=True)
    perm = np.arange(len(km))

    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        perm[ids] = rng.permutation(ids)

    matched = km[perm]

    m = km.shape[1]
    order = np.concatenate([
        rng.permutation(m),
        np.arange(m, len(rec))
    ])

    feedback = rec[np.ix_(order, order)]
    mod = dm[:, order[:m]]

    used = (km > 0).any(1)
    changed = (matched[used] != km[used]).any(1).mean()

    return shuffled, matched, feedback, mod, changed
