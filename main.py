
import torch
from connectome import load
from model import Memory


def odor(w, seed, noise=0):
    g = torch.Generator().manual_seed(seed)
    pn = torch.randperm(w.size(0), generator=g)[:16].clone()

    if noise:
        g.manual_seed(seed * 1000 + noise + 100000)
        pn[:4] = torch.randperm(w.size(0), generator=g)[:4]

    drive = w[pn].sum(0)
    x = torch.zeros(w.size(1))
    x[drive.topk(w.size(1) // 20).indices] = 1
    return x


def measure(s):
    a, b = s[::2], s[1::2]
    acc = ((a > b).float() + 0.5 * (a == b).float()).mean().item()
    margin = (a - b).mean().item()
    return acc, margin


def recall(m, x, cues):
    state = m.active.clone()
    scores = []

    for xi, ci in zip(x, cues):
        m.forget()
        m.remind(ci)
        scores.append(m.score(xi))

    m.active = state
    return torch.stack(scores)


def run(pn_kc, kc_mb, seed):
    ids = [100 * seed + i for i in range(20)]
    train = torch.stack([odor(pn_kc, i) for i in ids])
    cue = torch.stack([odor(pn_kc, i, 1) for i in ids])
    test = torch.stack([odor(pn_kc, i, 2) for i in ids])
    m = Memory(kc_mb)

    for _ in range(3):
        for x in train[::2]:
            m.learn(x)

    learned = measure(m.score(test))[0]
    m.forget()
    forgotten = measure(m.score(test))[0]
    recovered = recall(m, test, cue)
    wrong = recall(m, test, cue.roll(1, 0))

    hidden = m.silent.clone()
    m.lesion()
    targeted = recall(m, test, cue)

    m.silent = hidden.clone()
    m.lesion(seed)
    random = recall(m, test, cue)

    m.silent = m.s0.clone()
    erased = recall(m, test, cue)

    return [
        learned, forgotten, measure(recovered)[0],
        measure(wrong)[0], measure(erased)[0],
        measure(recovered)[1], measure(targeted)[1],
        measure(random)[1]
    ]


def main():
    pn_kc, kc_mb = load()
    pn_kc /= pn_kc.sum(0).clamp_min(1)
    results = {"real": [], "shuffled": []}

    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        shuffled = kc_mb[torch.randperm(len(kc_mb), generator=g)]

        for name, w in [("real", kc_mb), ("shuffled", shuffled)]:
            results[name].append(run(pn_kc, w, seed))

    print("\n            learned  forgotten  recovered  wrong cue  no trace")

    for name, runs in results.items():
        avg = torch.tensor(runs).mean(0)
        print(f"{name:<12}" + "".join(f"{v:10.1%}" for v in avg[:5]))
        print(f"  recovery margin: {avg[5]:.3f} | targeted lesion: {avg[6]:.3f} | random lesion: {avg[7]:.3f}")

    a, b = map(lambda k: torch.tensor(results[k])[:, 2], results)
    diff = a - b
    print(f"\nreal - shuffled recovery: {diff.mean():+.1%} +/- {1.96 * diff.std() / len(diff)**0.5:.1%}")


if __name__ == "__main__":
    main()
