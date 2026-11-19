
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



def masks(m, train, y, seed):
    d = m.scale * m.delta
    sign = 1 - 2 * y.float()
    support = ((train[0].T @ sign) * (d[:, 0] - d[:, 1])).clamp_min(0)

    freq = train[1].sum(0)
    overlap = (train[0] * train[1]).sum(0)
    ids = ((support > 0) & (freq > 0)).nonzero().flatten()

    q = torch.tensor([0.25, 0.5, 0.75])
    bins = (
        4 * torch.bucketize(support, torch.quantile(support[ids], q))
        + torch.bucketize(freq, torch.quantile(freq[ids], q))
    )

    g = torch.Generator().manual_seed(seed + 5000)
    target, matched = [], []

    for k in bins[ids].unique():
        pool = ids[bins[ids] == k]
        n = max(1, len(pool) // 10)

        target.append(pool[overlap[pool].topk(n).indices])
        matched.append(pool[torch.randperm(len(pool), generator=g)[:n]])

    target = torch.cat(target)
    matched = torch.cat(matched)
    n = len(target)

    important = ids[support[ids].topk(n).indices]
    random = ids[torch.randperm(len(ids), generator=g)[:n]]

    def mask(indices):
        x = torch.zeros(len(support), dtype=torch.bool)
        x[indices] = True
        return x

    budget = support * freq
    ratio = (budget[target].sum() /
             budget[matched].sum().clamp_min(1e-9)).item()

    return {
        "baseline": None,
        "targeted": mask(target),
        "matched": mask(matched),
        "important": mask(important),
        "random": mask(random)
    }, (n, ratio)


def run(w, train, test, y, seed):
    m = Memory(w)
    g = torch.Generator().manual_seed(seed)

    for _ in range(12):
        for i in torch.randperm(12, generator=g):
            m.learn(train[0, i], int(y[i]))

    before = (m.predict(test[0]) == y).float().mean()
    variants, diagnostic = masks(m, train, y, seed)
    results = {}

    for name, freeze in variants.items():
        t = Memory(w)
        t.delta = m.delta.clone()
        g = torch.Generator().manual_seed(seed + 10000)

        for _ in range(12):
            for i in torch.randperm(12, generator=g):
                t.learn(train[1, i], int(1 - y[i]), freeze)

        a = t.predict(test[0]) == y
        b = t.predict(test[1]) == 1 - y

        results[name] = torch.tensor([
            before,
            a.float().mean(),
            b.float().mean(),
            (a & b).float().mean()
        ])

    return results, diagnostic


def interval(x):
    return f"{x.mean():+.1%} +/- {1.96 * x.std() / len(x)**0.5:.1%}"


def main():
    pn_kc, kc_mb, names = load()
    pn_kc /= pn_kc.sum(0).clamp_min(1)

    pair = [names.index("MBON11"), names.index("MBON18")]
    real = kc_mb[:, pair]

    variants = ["baseline", "targeted", "matched", "important", "random"]
    results = {
        k: {v: [] for v in variants}
        for k in ["real", "shuffled"]
    }
    diagnostics = []

    for seed in range(50):
        train, test, y = patterns(pn_kc, seed)

        g = torch.Generator().manual_seed(seed + 2000)
        shuffled = real[torch.randperm(len(real), generator=g)]

        for name, w in [("real", real), ("shuffled", shuffled)]:
            out, diag = run(w, train, test, y, seed)

            for v in variants:
                results[name][v].append(out[v])

            if name == "real":
                diagnostics.append(diag)

    for name in results:
        print(f"\n{name} wiring")
        print("condition         before     old     new    both")

        for v in variants:
            a = torch.stack(results[name][v]).mean(0)
            print(f"{v:<17}" + " ".join(f"{x:7.1%}" for x in a))

        targeted = torch.stack(results[name]["targeted"])
        matched = torch.stack(results[name]["matched"])
        baseline = torch.stack(results[name]["baseline"])

        print("targeted - matched (both):",
              interval(targeted[:, 3] - matched[:, 3]))
        print("targeted - baseline (both):",
              interval(targeted[:, 3] - baseline[:, 3]))
        print("targeted - matched (new):",
              interval(targeted[:, 2] - matched[:, 2]))

    def gain(name):
        a = torch.stack(results[name]["targeted"])
        b = torch.stack(results[name]["matched"])
        return a[:, 3] - b[:, 3]

    print("\nextra targeted benefit from real wiring:",
          interval(gain("real") - gain("shuffled")))

    d = torch.tensor(diagnostics)
    print(f"\nprotected KCs: {d[:, 0].mean():.0f} on average")
    print(f"target/matched protection budget: {d[:, 1].mean():.2f}x")


if __name__ == "__main__":
    main()