
import torch
from connectome import load
from model import Memory


def make_odor(w, seed):
    rng = torch.Generator().manual_seed(seed)
    pn = torch.zeros(w.size(0))
    pn[torch.randperm(len(pn), generator=rng)[:max(1, len(pn) // 8)]] = 1

    drive = pn @ (w / w.sum(0).clamp_min(1))
    odor = torch.zeros_like(drive)
    odor[drive.topk(max(1, len(drive) // 20)).indices] = 1
    return odor


def main():
    pn_kc, kc_mb = load()
    a, b = make_odor(pn_kc, 1), make_odor(pn_kc, 2)

    rng = torch.Generator().manual_seed(3)
    shuffled = kc_mb[torch.randperm(kc_mb.size(0), generator=rng)]

    print(f"Odor overlap: {int((a * b).sum())}/{int(a.sum())} KCs")

    for name, w in [("Real", kc_mb), ("Shuffled", shuffled)]:
        m = Memory(w)
        before = (m.respond(a), m.respond(b))

        for _ in range(10):
            m.learn(a, reward=1)

        after = (m.respond(a), m.respond(b))
        print(f"\n{name} connectivity")
        print(f"Before: A={before[0]:.4f}, B={before[1]:.4f}")
        print(f"After:  A={after[0]:.4f}, B={after[1]:.4f}")


if __name__ == "__main__":
    main()
