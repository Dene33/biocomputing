"""Track (a): the teacher's own weights (78 -> 512 -> 256 -> 128 -> 23, ELU), edited in code so it works with the
dish's vestibular estimate. See log.md for the iterations."""

from pathlib import Path

import numpy as np

_T = dict(np.load(Path(__file__).parent / "teacher_weights.npz"))

# Input gains on the first layer's IMU columns (1 = the teacher's own weights).
GYRO_GAIN = np.array([1.5, 1.5, 1.5])
GRAV_GAIN = 1.0
# Front end: rotate the (late) gravity estimate forward by PRED_STEPS control steps with the gyro, dg/dt = g x w.
PRED_STEPS = 0
# Lead on gravity: g + GRAV_LEAD * (g - previous distinct sample), extrapolating the dish's late value.
GRAV_LEAD = 0.0
DT = 0.02


def write_weights():
    W = {k.replace("actor.", "l").replace(".weight", "_w").replace(".bias", "_b"): v.astype(np.float64).copy()
         for k, v in _T.items()}
    # Keep the standing operating point: gravity (0, 0, -1) feeds the first layer through column 5, so rescaling it
    # moves the bias by (1 - gain) * column * (-1).
    W["l0_b"] += (1.0 - GRAV_GAIN) * W["l0_w"][:, 5] * -1.0
    W["l0_w"][:, 0:3] *= GYRO_GAIN[None, :]
    W["l0_w"][:, 3:6] *= GRAV_GAIN
    W["state"] = {}
    return W


def reset(W):
    W["state"].clear()


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def front_end(W, vest):
    w = vest[0:3] / 0.25  # rad/s
    g = vest[3:6] + PRED_STEPS * DT * np.cross(vest[3:6], w)
    st = W["state"]
    if "g_last" not in st:
        st["g_last"], st["g_prev"] = vest[3:6].copy(), vest[3:6].copy()
    if np.any(vest[3:6] != st["g_last"]):  # a new sample from the dish
        st["g_prev"], st["g_last"] = st["g_last"], vest[3:6].copy()
    g = g + GRAV_LEAD * (st["g_last"] - st["g_prev"])
    return np.concatenate([vest[0:3], g / np.linalg.norm(g)])


def act(W, jpos, jvel, last_action, vest):
    vest = front_end(W, vest)
    x = np.concatenate([vest, np.zeros(3), jpos, jvel, last_action])
    h = elu(W["l0_w"] @ x + W["l0_b"])
    h = elu(W["l2_w"] @ h + W["l2_b"])
    h = elu(W["l4_w"] @ h + W["l4_b"])
    return W["l6_w"] @ h + W["l6_b"]
