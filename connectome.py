
from pathlib import Path
import pandas as pd
import torch
from flywire_gnn import FlyWireFAFB


def load():
    d = FlyWireFAFB().data
    path = Path.home() / ".cache/flywire_gnn/annotations_v783.tsv"
    a = pd.read_csv(path, sep="\t", usecols=[
        "root_id", "side", "cell_class", "cell_sub_class",
        "top_nt", "hemibrain_type"
    ])
    a = a[a.side == "right"]

    def ids(cls):
        b = a[a.cell_class == cls]
        if cls == "ALPN":
            b = b[(b.cell_sub_class == "uniglomerular") &
                  (b.top_nt == "acetylcholine")]
        return torch.tensor(sorted(b.root_id), dtype=torch.long)

    pn, kc = ids("ALPN"), ids("Kenyon_Cell")
    mb = a[(a.cell_class == "MBON") &
           a.hemibrain_type.isin(["MBON11", "MBON18"])]

    assert len(mb) == 2, "expected MBON11 and MBON18"
    mid = torch.tensor(sorted(mb.root_id), dtype=torch.long)
    names = mb.set_index("root_id").loc[mid.tolist(), "hemibrain_type"].tolist()

    u, v = d.edge_index

    def matrix(src, dst):
        mask = torch.isin(d.node_ids[u], src) & torch.isin(d.node_ids[v], dst)
        i = torch.searchsorted(src, d.node_ids[u[mask]])
        j = torch.searchsorted(dst, d.node_ids[v[mask]])
        w = torch.zeros(len(src), len(dst))
        w.index_put_((i, j), d.edge_attr[mask, 0].expm1(), accumulate=True)
        return w

    print(f"PNs: {len(pn)}, KCs: {len(kc)}, MBONs: {names}")
    return matrix(pn, kc), matrix(kc, mid), names
