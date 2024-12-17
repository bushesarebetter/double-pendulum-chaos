import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

G = 9.81
M1 = M2 = 1.0
L1 = L2 = 1.0


def derivs(s):
    # s[..., :] = (theta1, theta2, omega1, omega2); works on a single state or a stack of them
    t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    d = t1 - t2
    den = 2 * M1 + M2 - M2 * np.cos(2 * d)
    a1 = (-G * (2 * M1 + M2) * np.sin(t1) - M2 * G * np.sin(t1 - 2 * t2)
          - 2 * np.sin(d) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(d))) / (L1 * den)
    a2 = (2 * np.sin(d) * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(t1)
                           + w2**2 * L2 * M2 * np.cos(d))) / (L2 * den)
    return np.stack([w1, w2, a1, a2], axis=-1)


def rk4(s0, dt, steps):
    out = np.empty((steps + 1,) + np.shape(s0))
    s = out[0] = np.asarray(s0, dtype=float)
    for i in range(steps):
        k1 = derivs(s)
        k2 = derivs(s + 0.5 * dt * k1)
        k3 = derivs(s + 0.5 * dt * k2)
        k4 = derivs(s + dt * k3)
        s = out[i + 1] = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def energy(s):
    t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    T = (0.5 * (M1 + M2) * L1**2 * w1**2 + 0.5 * M2 * L2**2 * w2**2
         + M2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2))
    V = -(M1 + M2) * G * L1 * np.cos(t1) - M2 * G * L2 * np.cos(t2)
    return T + V








