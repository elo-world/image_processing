from PIL import Image
import numpy as np
import time


def median(src: np.ndarray, s: int) -> np.ndarray:
    h, w, _ = src.shape

    b_im = src.copy()

    for y in range(s, h - s):
        for x in range(s, w - s):
            s_box = src[y - s : y + s + 1, x - s : x + s + 1]
            b_im[y, x] = np.median(s_box, axis=(0, 1))

    return b_im.astype(np.uint8)


if __name__ == "__main__":
    im = Image.open("../images/snp.png")
    array = np.array(im)

    start_timer = time.time()

    m_level = int(input("Choisir le niveau : "))

    m_im = Image.fromarray(median(array, m_level))

    print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s")

    m_im.show()
