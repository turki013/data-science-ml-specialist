"""Regenerate the example frozen test set and print its canonical sha256.

The CSV is committed and static on purpose: it is NOT regenerated at test time (random generators are not
guaranteed identical across library versions). Run this only to (re)create the example data, then paste the
printed hash into FROZEN_TEST_SHA256 in project_hooks_example.py as a deliberate, reviewed change."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import pandas as pd  # noqa: E402

import project_hooks_example as h  # noqa: E402
from dstk.leakage import canonical_fingerprint  # noqa: E402

if __name__ == "__main__":
    df = h._simulate(80, h.SEED + 1000)[h.RAW + [h.TARGET]]
    h.FROZEN_TEST_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(h.FROZEN_TEST_PATH, index=False)
    back = pd.read_csv(h.FROZEN_TEST_PATH, float_precision="round_trip")
    print(canonical_fingerprint(back[h.RAW].assign(__y__=back[h.TARGET].to_numpy())))
