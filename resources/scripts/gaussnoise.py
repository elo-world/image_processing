from PIL import Image
import numpy as np
import time

def gaussnoise(src: np.ndarray, sd: float) -> np.ndarray:
    noise = np.random.randn(*src.shape) * sd
    noisy = src + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


if __name__ == "__main__":
    im = Image.open("../images/Lenna.png")
    array = np.array(im)

    start_timer = time.time()

    g_level = float(input("Choisir l'écart type du bruit (ex: 10) : "))

    g_im = Image.fromarray(gaussnoise(array, g_level))

    print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s")

    g_im.show()
