
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from connectome import load
from brain import Brain
from world import World
from agents import Linear, Tabular, controls


def choose(q, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(2))

    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


def episode(agent, world, c, l, rng, eps=0,
            learn=False, noise=0, blind=False):
    agent.reset()
    cue = 4 if blind else c

    ids = world.features[cue, l, 0, noise]
    q, r1 = agent.values(ids, (cue, l, 0))
    a = choose(q, eps, rng)

    node = 1 + a
    ids = world.features[cue, l, node, noise]
    q, r2 = agent.values(ids, (cue, l, node))
    b = choose(q, eps, rng)

    reward = world.reward(c, l, a, b)

    if learn:
        agent.learn([(r1, a), (r2, b)], reward)

    return float(reward > 0)


def probe(agent, world, contexts, seed,
          noisy=False, blind=False):
    rng = np.random.default_rng(seed)
    scores = []

    for c in contexts:
        for l in range(world.landmarks):
            for n in ([1, 2] if noisy else [0]):
                scores.append(episode(
                    agent, world, c, l, rng,
                    noise=n, blind=blind
                ))

    return np.mean(scores)


def run(agent, world, seed, episodes):
    rng = np.random.default_rng(seed)
    seen = set()

    initial = []
    returning = []
    retention = []
    learning = []

    for block, c in enumerate(world.schedule):
        scores = []
        repeated = c in seen

        for t in range(episodes):
            l = int(rng.integers(world.landmarks))
            eps = max(0.08, 0.35 * (1 - t / episodes))

            scores.append(episode(
                agent, world, c, l, rng,
                eps=eps, learn=True
            ))

        early = np.mean(scores[:24])

        if repeated:
            returning.append(early)
        else:
            initial.append(early)

        learning.append(np.mean(scores[-48:]))

        previous = sorted(seen - {c})

        if previous:
            retention.append(probe(
                agent, world, previous,
                seed + block + 10000
            ))

        seen.add(c)

    contexts = list(range(world.contexts))

    return np.array([
        np.mean(initial),
        np.mean(returning),
        np.mean(learning),
        np.mean(retention),
        probe(agent, world, contexts, seed + 20000),
        probe(agent, world, contexts, seed + 30000, noisy=True),
        probe(agent, world, contexts, seed + 40000, blind=True)
    ])


def interval(x):
    if len(x) < 2:
        return f"{x.mean():+.1%}"

    ci = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.1%} +/- {ci:.1%}"


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 240

    pk, km, rec, dm = load()
    results = {}
    changes = []

    for seed in range(seeds):
        world = World(pk, seed)

        shuffled, matched, feedback, mod, changed = controls(
            km, rec, dm, world.features, seed + 1000
        )
        changes.append(changed)

        models = {
            "real": Brain(km, rec, dm, seed),
            "shuffled": Brain(shuffled, rec, dm, seed),
            "matched": Brain(matched, rec, dm, seed),
            "rewired feedback": Brain(km, feedback, mod, seed),
            "no feedback": Brain(km, rec, dm, seed, "no feedback"),
            "no dopamine": Brain(km, rec, dm, seed, "no dopamine"),
            "no slow": Brain(km, rec, dm, seed, "no slow"),
            "persistent": Brain(km, rec, dm, seed, "persistent"),
            "linear": Linear(len(km)),
            "tabular": Tabular()
        }

        for name, agent in models.items():
            score = run(agent, world, seed + 5000, episodes)
            results.setdefault(name, []).append(score)

        print(f"seed {seed + 1}/{seeds} complete", flush=True)

    results = {k: np.stack(v) for k, v in results.items()}

    print("\nmodel              first  return   learn  retain   final   noisy   blind")

    for name, scores in results.items():
        print(f"{name:<19}" + " ".join(
            f"{v:6.1%}" for v in scores.mean(0)
        ))

    real = results["real"]

    print("\npaired comparisons (real - control)")

    for name, scores in results.items():
        if name == "real":
            continue

        print(f"\n{name}")
        print("retention:", interval(real[:, 3] - scores[:, 3]))
        print("final:    ", interval(real[:, 4] - scores[:, 4]))
        print("noisy:    ", interval(real[:, 5] - scores[:, 5]))

    print(f"\nactivity-matched rows changed: {np.mean(changes):.1%}")


if __name__ == "__main__":
    main()
