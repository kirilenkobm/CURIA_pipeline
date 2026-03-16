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
    if window_len <= 1:
        return signal
    kernel = np.ones(window_len) / window_len
    return np.convolve(signal, kernel, mode='same')
