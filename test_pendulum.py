import numpy as np

from pendulum import G, L1, energy, rk4


def test_energy_drift_small_dt():
    traj = rk4([2.0, 2.0, 0.0, 0.0], 1e-3, 10_000)
    e = energy(traj)
    assert np.max(np.abs(e - e[0])) / abs(e[0]) < 1e-5


def measured_freq(s0, dt=1e-3, steps=20_000):
    th1 = rk4(s0, dt, steps)[:, 0]
    up = np.where((th1[:-1] < 0) & (th1[1:] >= 0))[0]
    return 2 * np.pi / (np.diff(up).mean() * dt)


def test_small_angle_normal_modes():
    # equal masses and lengths: modes theta2 = +-sqrt(2) theta1, omega^2 = (2 -+ sqrt(2)) g / l
    a = 1e-3
    slow = measured_freq([a, np.sqrt(2) * a, 0.0, 0.0])
    fast = measured_freq([a, -np.sqrt(2) * a, 0.0, 0.0])
    assert abs(slow - np.sqrt((2 - np.sqrt(2)) * G / L1)) / slow < 0.01
    assert abs(fast - np.sqrt((2 + np.sqrt(2)) * G / L1)) / fast < 0.01
