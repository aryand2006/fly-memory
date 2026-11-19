
import torch
from connectome import load
from model import Memory


def encode(w, pn, ctx):
    drive = w[torch.cat([pn, ctx])].sum(0)
    x = torch.zeros(w.size(1))
    x[drive.topk(w.size(1) // 20).indices] = 1
    return x


def mutate(v, n, pool, g):
    available = pool[~torch.isin(pool, v)]
    replacement = available[torch.randperm(len(available), generator=g)[:n]]
    return torch.cat([replacement, v[n:]])


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

    plain = torch.stack([
        encode(w, p, torch.empty(0, dtype=torch.long))
        for p in noisy
    ])

    y = torch.randperm(12, generator=g) % 2
    return train, test, plain, y


def acc(pred, y):
    return (pred == y).float().mean().item()


def run(w, train, test, plain, y, seed):
    m = Memory(w)
    g = torch.Generator().manual_seed(seed)

    def learn(ctx, labels):
        for _ in range(12):
            for i in torch.randperm(12, generator=g):
                m.learn(train[ctx, i], int(labels[i]))

    learn(0, y)
    old_before = acc(m.predict(test[0]), y)

    learn(1, 1 - y)
    old_after = acc(m.predict(test[0]), y)
    new = acc(m.predict(test[1]), 1 - y)
    without_ctx = acc(m.predict(plain), y)

    flip = (m.predict(test[0]) != m.predict(test[1])).float().mean().item()

    def lesion(j):
        a = acc(m.predict(test[0], j), y)
        b = acc(m.predict(test[1], j), 1 - y)
        return (a + b) / 2

    return [
        old_before, old_after, new, without_ctx,
        flip, lesion(0), lesion(1)
    ]


def main():
    pn_kc, kc_mb, names = load()
    pn_kc /= pn_kc.sum(0).clamp_min(1)

    results = {"real": [], "shuffled": []}

    for seed in range(30):
        train, test, plain, y = patterns(pn_kc, seed)

        g = torch.Generator().manual_seed(seed + 1000)
        shuffled = kc_mb[torch.randperm(len(kc_mb), generator=g)]

        results["real"].append(run(kc_mb, train, test, plain, y, seed))
        results["shuffled"].append(run(shuffled, train, test, plain, y, seed))

    a = torch.tensor(results["real"])
    b = torch.tensor(results["shuffled"])

    metrics = [
        "old before switch", "old after switch",
        "new context", "without context",
        "context flips", f"without {names[0]}", f"without {names[1]}"
    ]

    print("\nmetric                   real    shuffled")
    for i, name in enumerate(metrics):
        print(f"{name:<24}{a[:, i].mean():6.1%}    {b[:, i].mean():6.1%}")

    for i, name in [(1, "old memory"), (2, "new memory")]:
        diff = a[:, i] - b[:, i]
        ci = 1.96 * diff.std() / len(diff) ** 0.5
        print(f"\n{name}, real - shuffled: {diff.mean():+.1%} +/- {ci:.1%}")


if __name__ == "__main__":
    main()
