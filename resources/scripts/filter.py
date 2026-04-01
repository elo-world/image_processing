from PIL import Image
import numpy as np
import math
import time

from resources.scripts.im_processing import ImProcessing


class Filter(ImProcessing):
    def blur(self, s: int) -> np.ndarray:

        padded = np.pad(self.src, ((s, s), (s, s), (0, 0)), mode="edge")
        b_im = self.src.copy()

        for y in range(self.h):
            for x in range(self.w):
                s_box = padded[y : y + 2 * s + 1, x : x + 2 * s + 1]
                b_im[y, x] = np.mean(s_box, axis=(0, 1))

        return b_im.astype(np.uint8)

    def gaussnoise(self, sd: float) -> np.ndarray:
        noise = np.random.randn(*self.src.shape) * sd
        noisy = self.src + noise
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)

    def saltnpepper(self, density: float) -> np.ndarray:
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
        n = math.floor(self.w * self.h * density / 2)
        dst = np.copy(self.src)

        x = np.random.randint(0, self.w, size=n)
        y = np.random.randint(0, self.h, size=n)

        for i in range(0, n):
            dst[y[i], x[i]] = [0] * 3

        x = np.random.randint(0, self.w, size=n)
        y = np.random.randint(0, self.h, size=n)

        for i in range(0, n):
            dst[y[i], x[i]] = [255] * 3

        return dst

    def median(self, s: int) -> np.ndarray:

        padded = np.pad(self.src, ((s, s), (s, s), (0, 0)), mode="edge")
        b_im = self.src.copy()

        for y in range(self.h):
            for x in range(self.w):
                s_box = padded[y : y + 2 * s + 1, x : x + 2 * s + 1]
                b_im[y, x] = np.median(s_box, axis=(0, 1))

        return b_im.astype(np.uint8)

    def menu(self, option: str) -> np.ndarray:
        match option:
            case "b":
                b_level = int(input("Choisir le niveau de flou : "))
                filtered_arr = self.blur(b_level)

            case "g":
                g_level = float(input("Choisir l'écart type du bruit (ex: 10) : "))
                filtered_arr = self.gaussnoise(g_level)

            case "snp":
                density = float(input("Choisir la densité du bruit : "))
                filtered_arr = self.saltnpepper(density)

            case "m":
                m_level = int(input("Choisir le niveau : "))
                filtered_arr = self.median(m_level)

            case _:
                filtered_arr = None

        return filtered_arr


if __name__ == "__main__":
    Filter().saltnpepper(0.1)
