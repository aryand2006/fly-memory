
import torch
from connectome import load
from model import Memory


def mutate(v, n, pool, g):
    available = pool[~torch.isin(pool, v)]
    return torch.cat([
        available[torch.randperm(len(available), generator=g)[:n]],
        v[n:]
    ])


def encode(w, pn, ctx):
    drive = w[torch.cat([pn, ctx])].sum(0)
    x = torch.zeros(w.size(1))
    x[drive.topk(w.size(1) // 20).indices] = 1
    return x


def patterns(w, seed):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(w.size(0), generator=g)
    ctx = order[:16].reshape(2, 8)
    pool = order[16:]

    odors = [
        pool[torch.randperm(len(pool), generator=g)[:16]]
        for _ in range(12)
    ]
    noisy = [mutate(p, 2, pool, g) for p in odors]
    noisy_ctx = [mutate(c, 1, pool, g) for c in ctx]

    train = torch.stack([
        torch.stack([encode(w, p, c) for p in odors])
        for c in ctx
    ])
    test = torch.stack([
        torch.stack([encode(w, p, c) for p in noisy])
        for c in noisy_ctx
    ])

    y = torch.randperm(12, generator=g) % 2
    return train, test, y


def accuracy(p, y):
    return (p == y).float().mean().item()


def both(m, test, y):
    return ((m.predict(test[0]) == y) &
            (m.predict(test[1]) == 1 - y)).float().mean().item()


def train_model(w, train, test, y, seed):
    m = Memory(w)
    g = torch.Generator().manual_seed(seed)

    for c in range(2):
        target = y if c == 0 else 1 - y
        for _ in range(12):
            for i in torch.randperm(12, generator=g):
                m.learn(train[c, i], int(target[i]))

        if c == 0:
            before = accuracy(m.predict(test[0]), y)

    a, b = m.predict(test[0]), m.predict(test[1])
    scores = torch.tensor([
        before,
        accuracy(a, y),
        accuracy(b, 1 - y),
        ((a == y) & (b == 1 - y)).float().mean().item()
    ])
    return m, scores


def shuffle(w, freq, seed, matched=False):
    g = torch.Generator().manual_seed(seed)

    if not matched:
        return w[torch.randperm(len(w), generator=g)]

    perm = torch.arange(len(w))
    for f in freq.unique(dim=0):
        ids = (freq == f).all(1).nonzero().flatten()
        perm[ids] = ids[torch.randperm(len(ids), generator=g)]

    return w[perm]


def lesions(m, train, test, y, seed):
    activity = train.mean((0, 1))
    ids = ((activity > 0) & (m.scale.sum(1) > 0)).nonzero().flatten()
    n = max(1, len(ids) // 10)

    importance = activity[ids] * m.scale[ids].sum(1)
    targeted = ids[importance.topk(n).indices]

    g = torch.Generator().manual_seed(seed + 2000)
    random = ids[torch.randperm(len(ids), generator=g)[:n]]

    saved = m.scale.clone()
    scores = []

    for cut in (targeted, random):
        m.scale[cut] = 0
        scores.append(both(m, test, y))
        m.scale = saved.clone()

    return scores


def matched_pairs(w, names):
    ref = [names.index("MBON11"), names.index("MBON18")]
    stats = torch.stack([w.sum(0), (w > 0).sum(0)], dim=1).float()
    used = set(ref)
    pairs = [("real", ref)]

    for k in range(2):
        pair = []
        for j in ref:
            dist = ((stats - stats[j]).abs() /
                    stats[j].clamp_min(1)).sum(1)

            for idx in used:
                dist[idx] = float("inf")

            pick = int(dist.argmin())
            pair.append(pick)
            used.add(pick)

        pairs.append((f"other pair {k + 1}", pair))

    return pairs


def main():
    pn_kc, kc_mb, names = load()
    pn_kc /= pn_kc.sum(0).clamp_min(1)
    pairs = matched_pairs(kc_mb, names)

    for label, pair in pairs:
        print(f"{label}: {names[pair[0]]}, {names[pair[1]]}")

    results = {k: [] for k in [
        "real", "shuffled", "activity matched",
        "other pair 1", "other pair 2"
    ]}
    cuts = []

    for seed in range(30):
        train, test, y = patterns(pn_kc, seed)
        freq = train.sum(1).T
        real = kc_mb[:, pairs[0][1]]

        if seed == 0:
            sh = shuffle(real, freq, seed + 1000, True)
            used = (real != 0).any(1)
            changed = (sh[used] != real[used]).any(1).float().mean()
            print(f"matched shuffle changed {changed:.1%} of connected KC rows")
            print("mbon pairs:", pairs)

        variants = [
            ("real", real),
            ("shuffled", shuffle(real, freq, seed + 1000)),
            ("activity matched", shuffle(real, freq, seed + 1000, True)),
            ("other pair 1", kc_mb[:, pairs[1][1]]),
            ("other pair 2", kc_mb[:, pairs[2][1]])
        ]

        for label, w in variants:
            m, scores = train_model(w, train, test, y, seed)
            results[label].append(scores)

            if label == "real":
                cuts.append(lesions(m, train, test, y, seed))

    print("\nmodel              before     old     new    both")
    for label, values in results.items():
        a = torch.stack(values).mean(0)
        print(f"{label:<18}" +
              " ".join(f"{v:7.1%}" for v in a))

    a = torch.stack(results["real"])

    for label in list(results)[1:]:
        b = torch.stack(results[label])
        diff = a[:, 3] - b[:, 3]
        ci = 1.96 * diff.std() / len(diff) ** 0.5
        print(f"real - {label} (both): {diff.mean():+.1%} +/- {ci:.1%}")

    cuts = torch.tensor(cuts).mean(0)
    print(f"\nreal both before lesions: {a[:, 3].mean():.1%}")
    print(f"targeted lesion: {cuts[0]:.1%}")
    print(f"random lesion:   {cuts[1]:.1%}")


if __name__ == "__main__":
    main()
