
import torch
from connectome import load
from model import Memory


def odor(w, seed, noise=0):
    g = torch.Generator().manual_seed(seed)
    pn = torch.randperm(w.size(0), generator=g)[:16].clone()

    if noise:
        g.manual_seed(seed * 1000 + noise + 100000)
        pn[:3] = torch.randperm(w.size(0), generator=g)[:3]

    drive = w[pn].sum(0)
    x = torch.zeros(w.size(1))
    x[drive.topk(w.size(1) // 20).indices] = 1
    return x


def accuracy(m, x, y):
    return (m.predict(x) == y[:, None]).float().mean().item()


def run(w, x, test, y, lr):
    m = Memory(w, lr)
    before = accuracy(m, test[:10], y[:10])

    for _ in range(2):
        for i in range(10):
            m.learn(x[i], int(y[i]))

    learned = accuracy(m, test[:10], y[:10])

    for _ in range(2):
        for i in range(10, 20):
            m.learn(x[i], int(y[i]))

    retained = accuracy(m, test[:10], y[:10])
    new = accuracy(m, test[10:], y[10:])
    return before, learned, retained, new


def main():
    pn_kc, kc_mb = load()
    pn_kc /= pn_kc.sum(0).clamp_min(1)

    results = {"real": [], "shuffled": [], "no learning": []}

    for seed in range(30):
        ids = [seed * 100 + i for i in range(20)]
        x = torch.stack([odor(pn_kc, s) for s in ids])
        test = torch.stack([
            torch.stack([odor(pn_kc, s, j) for j in range(1, 4)])
            for s in ids
        ])
        y = torch.arange(20) % 2

        g = torch.Generator().manual_seed(seed)
        shuffled = kc_mb[torch.randperm(kc_mb.size(0), generator=g)]

        for name, w, lr in [
            ("real", kc_mb, 0.15),
            ("shuffled", shuffled, 0.15),
            ("no learning", kc_mb, 0),
        ]:
            results[name].append(run(w, x, test, y, lr))

    print("\n             before  learned  retained  new")
    for name, values in results.items():
        avg = torch.tensor(values).mean(0) * 100
        print(f"{name:<12} " + "  ".join(f"{v:5.1f}%" for v in avg))

    real = torch.tensor(results["real"])
    shuffled = torch.tensor(results["shuffled"])
    diff = real[:, 2] - shuffled[:, 2]
    se = diff.std() / len(diff) ** 0.5
    print(f"\nretention difference: {diff.mean():+.1%} +/- {1.96 * se:.1%}")


if __name__ == "__main__":
    main()
