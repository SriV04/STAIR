from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "heterograph"))
from heterograph import HGraph  # noqa: E402
from IR.sched_ir import schema as schema_m  # noqa: E402


def main() -> None:
    g = HGraph(
        vinit=schema_m.vinit_sched,
        einit=schema_m.einit_sched,
        ginit=schema_m.ginit_sched,
    )
    u = g.add_vx()
    v = g.add_vx()
    print("node keys:")
    for key in sorted(g.pmap[u].keys()):
        print(f"  {key}")
    e = g.add_edge(u, v)[0]
    print("\nedge keys:")
    for key in sorted(g.pmap[e].keys()):
        print(f"  {key}")
    print("\ngraph keys:")
    for key in sorted(g.pmap.keys()):
        if isinstance(key, str):
            print(f"  {key}")


if __name__ == "__main__":
    main()
