
import torch
from connectome import load
from model import Memory


def mutate(pn, n, seed, size):
    g = torch.Generator().manual_seed(seed)
    choices = torch.randperm(size, generator=g)
    replacements = choices[~torch.isin(choices, pn)][:n]
    out = pn.clone()
    out[:n] = replacements
    return out


def odor(w, pn):
    drive = w[pn].sum(0)
    x = torch.zeros(w.size(1))
    x[drive.topk(w.size(1) // 20).indices] = 1
    return x


def accuracy(pred, y):
    return (pred == y).float().mean().item()


def run(pn_kc, w, seed):
    old, test, cue, new, new_test = [], [], [], [], []

    for i in range(12):
        s = seed * 100 + i
        g = torch.Generator().manual_seed(s)
        pn = torch.randperm(pn_kc.size(0), generator=g)[:16]

        q = mutate(pn, 4, s + 40000, pn_kc.size(0))

        old.append(odor(pn_kc, pn))
        test.append(odor(pn_kc, mutate(pn, 2, s + 10000, pn_kc.size(0))))
        cue.append(odor(pn_kc, mutate(pn, 3, s + 20000, pn_kc.size(0))))
        new.append(odor(pn_kc, q))
        new_test.append(odor(pn_kc, mutate(q, 2, s + 30000, pn_kc.size(0))))

    old, test, cue, new, new_test = [
        torch.stack(v) for v in (old, test, cue, new, new_test)
    ]

    y = torch.arange(12) % 2
    m = Memory(w)
    g = torch.Generator().manual_seed(seed)

    for _ in range(8):
        for i in torch.randperm(12, generator=g):
            m.learn(old[i], int(y[i]))

    learned = accuracy(m.predict(test), y)

    for _ in range(3):
        for i in torch.randperm(12, generator=g):
            m.learn(new[i], int(1 - y[i]))

    interfered = accuracy(m.predict(test), y)
    new_acc = accuracy(m.predict(new_test), 1 - y)

    recovered = accuracy(m.predict(test, cue), y)
    fast_cue = accuracy(m.predict(test, cue, "fast"), y)
    wrong = accuracy(m.predict(test, cue.roll(1, 0)), y)

    saved = m.slow.clone()

    m.lesion()
    targeted = accuracy(m.predict(test, cue), y)

    m.slow = saved.clone()
    m.lesion(seed)
    random = accuracy(m.predict(test, cue), y)

    m.slow.zero_()
    erased = accuracy(m.predict(test, cue), y)

    return [
        learned, interfered, new_acc, recovered,
        fast_cue, wrong, targeted, random, erased
    ]


def main():
    pn_kc, kc_mb = load()
    pn_kc /= pn_kc.sum(0).clamp_min(1)

    results = {"real": [], "shuffled": []}

    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        shuffled = kc_mb[torch.randperm(len(kc_mb), generator=g)]

        results["real"].append(run(pn_kc, kc_mb, seed))
        results["shuffled"].append(run(pn_kc, shuffled, seed))

    names = [
        "learned", "interfered", "new learning", "recovered",
        "fast cue", "wrong cue", "targeted lesion",
        "random lesion", "no slow trace"
    ]

    a = torch.tensor(results["real"])
    b = torch.tensor(results["shuffled"])

    print("\nmetric                real     shuffled")
    for i, name in enumerate(names):
        print(f"{name:<22}{a[:, i].mean():6.1%}    {b[:, i].mean():6.1%}")

    for name, values in results.items():
        v = torch.tensor(values)
        gain = v[:, 3] - v[:, 1]
        ci = 1.96 * gain.std() / len(gain) ** 0.5
        print(f"\n{name} recovery gain: {gain.mean():+.1%} +/- {ci:.1%}")

    diff = a[:, 3] - b[:, 3]
    ci = 1.96 * diff.std() / len(diff) ** 0.5
    print(f"real - shuffled recovery: {diff.mean():+.1%} +/- {ci:.1%}")

    gain = (a[:, 3] - a[:, 1]) - (b[:, 3] - b[:, 1])
    ci = 1.96 * gain.std() / len(gain) ** 0.5
    print(f"extra recovery from real wiring: {gain.mean():+.1%} +/- {ci:.1%}")



if __name__ == "__main__":
    main()
