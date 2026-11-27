
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from scipy.stats import t as student
from connectome import load
from brain import Brain, PARAMS, LO, HI, DEFAULT
from world import Codebook, World, run, transfer
from agents import Linear, Associative, architectures


def objective(scores):
    return (
        .5 * scores[0] +
        .25 * scores[2] +
        .25 * scores[4]
    )


def make(name, rule, circuits, world, mode=None):
    if name == "linear":
        return Linear(circuits["real"][0].g.shape[0], rule[0])

    if name == "assoc":
        return Associative(rule[0])

    options = circuits[name]
    circuit = options[world.seed % len(options)]

    mode = mode or ("single" if name == "single" else "full")

    return Brain(circuit, rule, mode)


def evaluate(name, rule, circuits, worlds, episodes):
    return np.mean([
        objective(run(
            make(name, rule, circuits, w),
            w, rule[-1], episodes
        ))
        for w in worlds
    ])


def evolve(name, circuits, train, val, generations,
           population, episodes):

    rng = np.random.default_rng(42)

    center = (DEFAULT - LO) / (HI - LO)
    sigma = np.full(len(center), .25)

    champion = center.copy()
    best_rule = DEFAULT.copy()
    best_val = -np.inf

    for gen in range(generations):
        candidates = np.vstack([
            champion,
            center,
            center + sigma * rng.normal(
                size=(population - 2, len(center))
            )
        ]).clip(0, 1)

        rules = LO + candidates * (HI - LO)

        scores = np.array([
            evaluate(name, rule, circuits, train, episodes)
            for rule in rules
        ])

        order = scores.argsort()[::-1]
        elite = candidates[order[:max(2, population // 4)]]

        champion = candidates[order[0]].copy()
        center = elite.mean(0)
        sigma = elite.std(0).clip(.07, .35)

        winner = rules[order[0]]
        v = evaluate(name, winner, circuits, val, episodes)

        if v > best_val:
            best_val = v
            best_rule = winner.copy()

        print(
            f"{name:<9} gen {gen + 1}/{generations} "
            f"train={scores.max():.1%} "
            f"val={v:.1%} best={best_val:.1%}",
            flush=True
        )

    return best_rule


def tune_baseline(name, circuits, train, val, episodes):
    candidates = [
        np.array([lr, eps])
        for lr in (.1, .3, .6, 1.)
        for eps in (.05, .15, .3, .5)
    ]

    scores = [
        evaluate(name, r, circuits, train, episodes)
        for r in candidates
    ]

    order = np.argsort(scores)[-4:]

    return max(
        [candidates[i] for i in order],
        key=lambda r: evaluate(
            name, r, circuits, val, episodes
        )
    )


def interval(x):
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return f"{x.mean():+.1%}"

    ci = (
        student.ppf(.975, len(x) - 1) *
        x.std(ddof=1) / np.sqrt(len(x))
    )

    return f"{x.mean():+.1%} +/- {ci:.1%}"


def conditions(rules):
    out = {
        name: (name, rule, None)
        for name, rule in rules.items()
    }

    out["initial"] = ("real", DEFAULT, None)

    for mode in [
        "no feedback", "no dopamine",
        "no state", "frozen"
    ]:
        out[mode] = ("real", rules["real"], mode)

    return out


def benchmark(circuits, rules, worlds, episodes, title):
    results = {}

    for label, (name, rule, mode) in conditions(rules).items():
        results[label] = np.stack([
            run(
                make(name, rule, circuits, w, mode),
                w, rule[-1], episodes
            )
            for w in worlds
        ])

    print(f"\n{title}")
    print(
        "model              online return retain "
        " final  noisy   long  blind    rev"
    )

    for name, scores in results.items():
        avg = [
            v[np.isfinite(v)].mean()
            if np.isfinite(v).any() else np.nan
            for v in scores.T
        ]

        print(f"{name:<19}" + " ".join(
            f"{v:6.1%}" for v in avg
        ))

    print("\nreal - control")

    real = results["real"]

    for name, scores in results.items():
        if name == "real":
            continue

        print(f"\n{name}")

        for i, metric in [
            (0, "online"),
            (2, "retention"),
            (4, "noisy"),
            (5, "long delay")
        ]:
            print(f"{metric:<12}", interval(
                real[:, i] - scores[:, i]
            ))

    return results


def memory_test(circuits, rules, worlds):
    print("\ntransfer: associative memory")
    print("model             1 write   noisy  3 writes   noisy")

    results = {}

    for name, rule in rules.items():
        results[name] = np.stack([
            transfer(make(name, rule, circuits, w), w)
            for w in worlds
        ])

        print(f"{name:<18}" + " ".join(
            f"{v:8.1%}" for v in results[name].mean(0)
        ))

    print("\nreal - associative baseline")
    diff = results["real"] - results["assoc"]
    print("one-shot noisy:", interval(diff[:, 1]))
    print("three-shot noisy:", interval(diff[:, 3]))


def main():
    args = [int(x) for x in sys.argv[1:]]
    generations, population, seeds, episodes = (
        args + [5, 12, 8, 48][len(args):]
    )

    assert generations >= 1
    assert population >= 4
    assert seeds >= 1 and episodes >= 1

    pk, km, rec, dm = load()

    code = Codebook(pk, 1)
    circuits = architectures(km, rec, dm, code)

    train = [World(code, s) for s in range(10, 13)]
    val = [World(code, s) for s in range(100, 102)]

    rules = {}

    for name in circuits:
        rules[name] = evolve(
            name, circuits, train, val,
            generations, population, episodes
        )

    for name in ("linear", "assoc"):
        rules[name] = tune_baseline(
            name, circuits, train, val, episodes
        )

    np.savez("rules.npz", **rules)

    print("\ndiscovered learning rules")

    for name in circuits:
        print(f"\n{name}")

        for key, value in zip(PARAMS, rules[name]):
            print(f"{key:<16} {value:.4f}")

    for name in ("linear", "assoc"):
        print(f"\n{name}: lr={rules[name][0]:.3f}, "
              f"eps={rules[name][1]:.3f}")

    test = [
        World(code, 1000 + i)
        for i in range(seeds)
    ]

    novel = [
        World(Codebook(pk, 9000 + i), 2000 + i)
        for i in range(seeds)
    ]

    benchmark(
        circuits, rules, test, episodes,
        "held-out reward mappings"
    )

    benchmark(
        circuits, rules, novel, episodes,
        "held-out sensory patterns"
    )

    memory_test(circuits, rules, novel)


if __name__ == "__main__":
    main()
