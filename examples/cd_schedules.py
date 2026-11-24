"""Print three CD frame orders using traceable IDs in place of recorded I/Q."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xfr.schedules import cd_blocks
from xfr.representation import cross_frame_rows


if __name__ == "__main__":
    # Each receiver supplies eight frames from one transmitter; L=3 here.
    domains = [100*rx + np.arange(8)[:, None] + 1j*np.arange(3) for rx in range(3)]
    for strategy in ("round_robin", "short_runs", "domain_groups"):
        physical = cd_blocks(domains, 12, strategy, chunk_size=2)
        rows = cross_frame_rows(physical)
        assert np.array_equal(rows[0, 1], physical[0, :, 1])
        print(f"{strategy:13s} {physical[0, :, 0].real.astype(int).tolist()}  XFR {rows.shape}")
