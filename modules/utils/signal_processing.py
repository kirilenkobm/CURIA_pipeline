#!/usr/bin/env python3
"""Shared signal processing utilities for island scanning."""

import numpy as np


def smooth_signal(signal: np.ndarray, window_len: int = 5) -> np.ndarray:
    """
    Apply box filter smoothing to a 1D signal.

    Args:
        signal: Input 1D numpy array
        window_len: Size of the smoothing window (must be positive)

    Returns:
        Smoothed signal with same shape as input
    """
    # np.convolve(mode='same') returns length max(len(signal), window_len), so a
    # signal shorter than the window would come back LONGER than the input (breaking
    # the 1:1 mapping to window positions downstream). Clamp the window to the signal
    # length to keep the output the same shape as the input. This happens for short
    # transcripts whose window count is < window_len (common with the larger RiNALMo
    # window/stride, e.g. a ~220 nt spliced gene -> 3 windows).
    n = len(signal)
    window_len = min(window_len, n)
    if window_len <= 1:
        return signal
    kernel = np.ones(window_len) / window_len
    return np.convolve(signal, kernel, mode='same')
