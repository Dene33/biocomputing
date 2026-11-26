from pathlib import Path

import numpy as np

_T = dict(np.load(Path(__file__).parent / "teacher_weights.npz"))

GYRO_GAIN = np.array([1.5, 1.6, 1.65])
GRAV_GAIN = np.array([1.59, 1.43, 1.0])
PRED_STEPS = 0
GRAV_LEAD = 0.0
DT = 0.02


def write_weights():
    W = {k.replace("actor.", "l").replace(".weight", "_w").replace(".bias", "_b"): v.astype(np.float64).copy()
         for k, v in _T.items()}
    W["l0_b"] += (1.0 - GRAV_GAIN[2]) * W["l0_w"][:, 5] * -1.0
    W["l0_w"][:, 0:3] *= GYRO_GAIN[None, :]
    W["l0_w"][:, 3:6] *= GRAV_GAIN[None, :]
    W["state"] = {}
    return W


def reset(W):
    W["state"].clear()


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def front_end(W, vest):
    w = vest[0:3] / 0.25
    g = vest[3:6] + PRED_STEPS * DT * np.cross(vest[3:6], w)
    st = W["state"]
    if "g_last" not in st:
        st["g_last"], st["g_prev"] = vest[3:6].copy(), vest[3:6].copy()
    if np.any(vest[3:6] != st["g_last"]):
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
