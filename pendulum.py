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


def lyapunov_fit(t, sep, d_max=1e-1):
    # fit log-separation only while it is still growing exponentially (before it saturates)
    stop = np.argmax(sep > d_max) if np.any(sep > d_max) else len(t)
    lam, c = np.polyfit(t[1:stop], np.log(sep[1:stop]), 1)
    return lam, c, t[stop - 1]


def render_gif(traj, dt, path, fps=25, size=360):
    every = round(1 / (fps * dt))
    scale = size / (2.2 * (L1 + L2))
    cx = cy = size // 2
    colors = [(30, 110, 220), (230, 80, 40)]
    trails = [[], []]
    frames = []
    for i in range(0, len(traj), every):
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        for k, (t1, t2, _, _) in enumerate(traj[i]):
            x1, y1 = cx + scale * L1 * np.sin(t1), cy + scale * L1 * np.cos(t1)
            x2, y2 = x1 + scale * L2 * np.sin(t2), y1 + scale * L2 * np.cos(t2)
            trails[k] = (trails[k] + [(x2, y2)])[-40:]
            if len(trails[k]) > 1:
                draw.line(trails[k], fill=tuple(c // 2 + 127 for c in colors[k]), width=1)
            draw.line([(cx, cy), (x1, y1), (x2, y2)], fill=colors[k], width=3)
            for x, y in ((x1, y1), (x2, y2)):
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=colors[k])
        draw.text((8, 8), f"t = {i * dt:5.1f} s", fill="black")
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t-end", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args()

    s0 = np.array([[2.0, 2.0, 0.0, 0.0], [2.0 + args.eps, 2.0, 0.0, 0.0]])
    steps = int(round(args.t_end / args.dt))
    traj = rk4(s0, args.dt, steps)
    t = np.arange(steps + 1) * args.dt

    sep = np.linalg.norm(traj[:, 1] - traj[:, 0], axis=1)
    lam, c, t_fit = lyapunov_fit(t, sep)
    e = energy(traj[:, 0])
    print(f"relative energy drift: {np.max(np.abs(e - e[0])) / abs(e[0]):.2e}")
    print(f"Lyapunov exponent estimate: {lam:.2f} 1/s (fit over t < {t_fit:.1f} s)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(t, sep, lw=1, label="phase-space separation")
    ax.semilogy(t[t <= t_fit], np.exp(c + lam * t[t <= t_fit]), "k--",
                label=f"fit: $\\lambda \\approx {lam:.2f}$ s$^{{-1}}$")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|Δstate|")
    ax.set_title(f"Two double pendulums started {args.eps:g} rad apart")
    ax.legend()
    fig.tight_layout()
    fig.savefig("separation.png", dpi=120)

    render_gif(traj, args.dt, "pendulums.gif")
    print("wrote separation.png, pendulums.gif")


if __name__ == "__main__":
    main()
