
import numpy as np


class World:
    contexts = 4
    landmarks = 6
    schedule = [0, 1, 2, 3, 0, 2, 1, 3]

    def __init__(self, pk, seed):
        rng = np.random.default_rng(seed)

        self.goals = np.stack([
            rng.permutation(np.arange(self.landmarks) % 4)
            for _ in range(self.contexts)
        ])

        order = rng.permutation(len(pk))
        cues = order[:32].reshape(4, 8)
        landmarks = order[32:80].reshape(6, 8)
        nodes = order[80:104].reshape(3, 8)

        k = pk.shape[1] // 20

        self.features = np.zeros(
            (5, 6, 3, 3, k), dtype=np.int32
        )

        def corrupt(ids, pool):
            available = np.setdiff1d(pool, ids)
            return np.concatenate([
                ids[2:],
                rng.choice(available, 2, replace=False)
            ])

        for c in range(5):
            for l in range(6):
                for noise in range(3):
                    lm = landmarks[l]
                    cue = cues[c] if c < 4 else np.array([], dtype=int)

                    if noise:
                        lm = corrupt(lm, landmarks.flatten())
                        if len(cue):
                            cue = corrupt(cue, cues.flatten())

                    for node in range(3):
                        drive = (
                            pk[lm].sum(0) +
                            pk[nodes[node]].sum(0)
                        )

                        if len(cue):
                            decay = 0.8 if node == 0 else 0.64
                            drive += decay * pk[cue].sum(0)

                        self.features[c, l, node, noise] = np.sort(
                            np.argpartition(drive, -k)[-k:]
                        )

    def reward(self, context, landmark, a, b):
        return 1.0 if 2 * a + b == self.goals[context, landmark] else -0.25
