from PIL import Image
import numpy as np
import time

def blur(src: np.ndarray, s: int) -> np.ndarray:
    h, w, _ = src.shape

    b_im = src.copy()

    for y in range(s, h - s):
        for x in range(s, w - s):
            s_box = src[y - s : y + s + 1, x - s : x + s + 1]
            b_im[y, x] = np.mean(s_box, axis=(0, 1))

    return b_im.astype(np.uint8)


if __name__ == "__main__":
    im = Image.open("../images/Lenna.png")
    array = np.array(im)

    start_timer = time.time()

    b_level = int(input("Choisir le niveau de flou : "))

    b_im = Image.fromarray(blur(array, b_level))

    print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s")

    b_im.show()
