import numpy as np

KERNELS = {
    "identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    "blur": np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9,
    "gaussian": np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16,
    "sharpness": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    "edge_dectection_1": np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]),
    "edge_dectection_2": np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]),
    "sobel_x": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    "sobel_y": np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]),
}
