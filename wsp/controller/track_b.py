import numpy as np

# Action indices (policy order) and mirror signs: the right leg's pitch axes point the other way.
L_HIP_P, L_HIP_R, L_ANK_P, L_ANK_R = 0, 1, 4, 5
R_HIP_P, R_HIP_R, R_ANK_P, R_ANK_R = 6, 7, 10, 11

K_ANK_P = -8.0
K_HIP_P = 0.0
K_ANK_R = 0.0
K_HIP_R = 0.0
D_P = 0.0
D_R = 0.0


def write_weights():
    G = np.zeros((23, 6))
    G[L_ANK_P, 3], G[R_ANK_P, 3] = K_ANK_P, -K_ANK_P
    G[L_HIP_P, 3], G[R_HIP_P, 3] = K_HIP_P, -K_HIP_P
    G[L_ANK_R, 4], G[R_ANK_R, 4] = K_ANK_R, K_ANK_R
    G[L_HIP_R, 4], G[R_HIP_R, 4] = K_HIP_R, K_HIP_R
    G[L_ANK_P, 1], G[R_ANK_P, 1] = D_P, -D_P
    G[L_ANK_R, 0], G[R_ANK_R, 0] = D_R, D_R
    return {"G": G, "stand": np.array([0, 0, 0, 0, 0, -1.0])}


def act(W, jpos, jvel, last_action, vest):
    return W["G"] @ (vest - W["stand"])
