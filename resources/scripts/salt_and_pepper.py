from PIL import Image
import numpy as np
import math
import time


def saltnpepper(src, density):
    """
    Add a Salt and Pepper noise.
    This function alters the
    value of certain pixels by
    changing them to black or
    white.
    Parameters
    ----------
    src (numpy.ndarray, float): input
    matrix of floats (0., 1.)
    density (float): noise density
    coefficient.
    Returns
    -------
    dst: (numpy.ndarray, float): matrix
    with added salt and pepper
    noise.
    """
    h = src.shape[0]
    w = src.shape[1]
    n = math.floor(w * h * density / 2)
    dst = np.copy(src)

    x = np.random.randint(0, w, size=n)
    y = np.random.randint(0, h, size=n)

    for i in range(0, n):
        dst[y[i], x[i]] = [0] * 3

    x = np.random.randint(0, w, size=n)
    y = np.random.randint(0, h, size=n)

    for i in range(0, n):
        dst[y[i], x[i]] = [255] * 3

    return dst


if __name__ == "__main__":
    im = Image.open("../images/Lenna.png")
    array = np.array(im)

    start_timer = time.time()

    density = float(input("Choisir la densité du bruit : "))

    snp_im = Image.fromarray(saltnpepper(array, density))

    print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s")

    snp_im.show()
