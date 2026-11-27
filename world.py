
import numpy as np


class Codebook:
    def __init__(self, pk, seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pk))
        k = pk.shape[1] // 20

        def encode(pn):
            z = pk[pn].sum(0)
            return np.sort(np.argpartition(z, -k)[-k:])

        def patterns(groups):
            result = []

            for pn in groups:
                variants = [encode(pn)]

                for _ in range(2):
                    pool = np.setdiff1d(order, pn)
                    q = pn.copy()
                    q[:2] = rng.choice(pool, 2, replace=False)
                    variants.append(encode(q))

                result.append(variants)

            return np.array(result, dtype=np.int32)

        self.cue = patterns(order[:32].reshape(4, 8))
        self.key = patterns(order[32:88].reshape(8, 7))
        self.dist = patterns(order[88:112].reshape(4, 6))

    def frequency(self, n):
        return np.column_stack([
            *[
                np.bincount(c[0], minlength=n)
                for c in self.cue
            ],
            np.bincount(self.key[:, 0].ravel(), minlength=n),
            np.bincount(self.dist[:, 0].ravel(), minlength=n)
        ])


class World:
    schedule = [0, 1, 2, 3, 0, 2, 1, 3]

    def __init__(self, code, seed):
        self.code = code
        self.seed = seed
        self.reversal = bool(seed % 2)

        rng = np.random.default_rng(seed)

        self.goals = np.stack([
            rng.permutation(np.arange(8) % 4)
            for _ in range(4)
        ])

        self.shift = rng.integers(1, 4, size=(4, 8))

    def changed(self, c, seen):
        return self.reversal and c in seen and c in (0, 2)


def choose(q, eps, rng):
    u, v = rng.random(2)

    if u < eps:
        return int(v * len(q))

    best = np.flatnonzero(q == q.max())
    return int(best[int(v * len(best))])


def trial(agent, world, goals, c, k, rng,
          eps=0, learn=False, noise=False,
          delay=2, blind=False, forced=None):

    code = world.code
    agent.reset()

    cn = int(rng.integers(1, 3)) if noise else 0
    kn = int(rng.integers(1, 3)) if noise else 0

    if not blind:
        agent.pulse(code.cue[c, cn])
    else:
        agent.pulse()

    for i in range(delay):
        d = int(rng.integers(4))

        if i % 2 == 0:
            agent.pulse(code.dist[d, 0], strength=.15)
        else:
            agent.pulse()

    q, record = agent.decide(code.key[k, kn])
    a = choose(q, eps, rng)

    if forced is not None:
        a = forced

    correct = a == goals[c, k]
    reward = 1.0 if correct else -1.0

    agent.pulse()
    agent.pulse()

    if learn:
        agent.learn(record, a, reward)

    return float(correct)


def probe(agent, world, goals, contexts, seed,
          noise=False, delay=2, blind=False):

    rng = np.random.default_rng(seed)

    return np.mean([
        trial(agent, world, goals, c, k, rng,
              noise=noise, delay=delay, blind=blind)
        for c in contexts
        for k in range(8)
    ])


def run(agent, world, eps, episodes):
    rng = np.random.default_rng(world.seed + 10000)
    goals = world.goals.copy()

    seen = set()
    online, returns, retained, reversal = [], [], [], []

    for block, c in enumerate(world.schedule):
        changed = world.changed(c, seen)

        if changed:
            goals[c] = (goals[c] + world.shift[c]) % 4

        if c in seen and not changed:
            returns.append(probe(
                agent, world, goals, [c],
                world.seed + block + 20000
            ))

        scores = []

        for _ in range(episodes):
            k = int(rng.integers(8))
            scores.append(trial(
                agent, world, goals, c, k, rng,
                eps=eps, learn=True
            ))

        online.extend(scores)

        if changed:
            reversal.append(np.mean(scores[:12]))

        previous = sorted(seen - {c})

        if previous:
            retained.append(probe(
                agent, world, goals, previous,
                world.seed + block + 30000
            ))

        seen.add(c)

    def final(**kwargs):
        return probe(
            agent, world, goals, range(4),
            world.seed + 40000, **kwargs
        )

    return np.array([
        np.mean(online),
        np.mean(returns),
        np.mean(retained),
        final(),
        final(noise=True),
        final(delay=8),
        final(blind=True),
        np.mean(reversal) if reversal else np.nan
    ])


def transfer(agent, world):
    rng = np.random.default_rng(world.seed + 50000)
    goals = world.goals.copy()
    result = []

    for r in range(3):
        for c in rng.permutation(4):
            for k in rng.permutation(8):
                trial(
                    agent, world, goals, c, k, rng,
                    learn=True, forced=int(goals[c, k])
                )

        if r in (0, 2):
            result.extend([
                probe(agent, world, goals, range(4),
                      world.seed + 60000),
                probe(agent, world, goals, range(4),
                      world.seed + 70000, noise=True)
            ])

    return np.array(result)
