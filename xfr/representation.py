"""Convert blocks of repeated references to position-wise cross-frame signals."""

import numpy as np


def cross_frame_rows(blocks):
    """Return [block, position, frame, ...] as a view of [block, frame, position, ...].

    Accept complex arrays and arrays with a final I/Q axis of length two.
    Each row retains the measurements of one sample position across the block.
    """
    blocks = np.asarray(blocks)
    if blocks.ndim not in (3, 4) or (blocks.ndim == 4 and blocks.shape[-1] != 2):
        raise ValueError("Expected [B, m, L] or [B, m, L, 2].")
    return blocks.swapaxes(1, 2)


def contiguous_blocks(references, m):
    """Take full consecutive SD blocks from one transmitter and one recording.

    A trailing group shorter than m is discarded. Call once per recording to
    preserve its boundary. The input must already be in acquisition order.
    """
    references = np.asarray(references)
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer.")
    if references.ndim not in (2, 3) or (references.ndim == 3 and references.shape[-1] != 2):
        raise ValueError("Expected [frames, L] or [frames, L, 2].")
    n_blocks = len(references) // m
    return references[:n_blocks*m].reshape(n_blocks, m, *references.shape[1:])
