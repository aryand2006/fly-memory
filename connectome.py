
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from flywire_gnn import FlyWireFAFB


def load():
    d = FlyWireFAFB().data
    path = Path.home() / ".cache/flywire_gnn/annotations_v783.tsv"

    a = pd.read_csv(path, sep="\t", low_memory=False)
    a = a[a.side == "right"]

    pn = a[
        (a.cell_class == "ALPN") &
        (a.cell_sub_class == "uniglomerular") &
        (a.top_nt == "acetylcholine")
    ]
    kc = a[a.cell_class == "Kenyon_Cell"]
    mb = a[a.cell_class == "MBON"]
    dan = a[a.cell_class == "DAN"]

    groups = [pn, kc, mb, dan]
    sizes = [len(g) for g in groups]
    p, k, m, n = sizes

    cells = pd.concat(groups)
    ids = cells.root_id.to_numpy()

    loc = pd.Index(d.node_ids.numpy()).get_indexer(ids)
    assert (loc >= 0).all()

    lookup = torch.full((len(d.node_ids),), -1)
    lookup[torch.from_numpy(loc)] = torch.arange(len(ids))

    u, v = lookup[d.edge_index]
    keep = (u >= 0) & (v >= 0)

    w = torch.zeros(len(ids), len(ids))
    w.index_put_(
        (u[keep], v[keep]),
        d.edge_attr[keep, 0].expm1().float(),
        accumulate=True
    )
    w = w.numpy()

    pk = w[:p, p:p+k].copy()
    km = w[p:p+k, p+k:p+k+m].copy()
    rec = w[p+k:, p+k:].copy()
    dm = rec[m:, :m].copy()

    pk /= np.maximum(pk.sum(0), 1)
    dm /= np.maximum(dm.sum(0), 1)

    nt = cells.top_nt.iloc[p+k:].to_numpy()
    sign = np.where(np.isin(nt, ["gaba", "glutamate"]), -1, 1)

    rec *= sign[:, None]
    rec /= np.maximum(np.abs(rec).sum(0), 1)

    print(f"PNs: {p}, KCs: {k}, MBONs: {m}, DANs: {n}")
    print(f"KC -> MBON connections: {np.count_nonzero(km):,}")
    print(f"recurrent connections: {np.count_nonzero(rec):,}")
    print(f"DAN -> MBON connections: {np.count_nonzero(dm):,}")

    return pk, km, rec, dm
