import numpy as np
import math
from PIL import Image


from resources.scripts.im_processing import ImProcessing, auto_timer


@auto_timer
class Filter(ImProcessing):
    def blur(self, s: int) -> np.ndarray:

        padded = np.pad(self.src, ((s, s), (s, s), (0, 0)), mode="edge")
        b_im = np.zeros_like(self.src, dtype=np.float32)

        for y in range(self.h):
            for x in range(self.w):
                s_box = padded[y : y + 2 * s + 1, x : x + 2 * s + 1]
                b_im[y, x] = np.mean(s_box, axis=(0, 1))

        return np.clip(b_im, 0, 255).astype(dtype=np.uint8)

    def hblur(self, src: np.ndarray, s: int) -> np.ndarray:
        padded = np.pad(src, ((0, 0), (s, s), (0, 0)), mode="edge")
        hb_im = np.zeros_like(src, dtype=np.float32)

        for i in range(2 * s + 1):
            hb_im += padded[:, i : i + src.shape[1], :]

        hb_im /= 2 * s + 1
        return np.clip(hb_im, 0, 255).astype(np.uint8)

    def vblur(self, src: np.ndarray, s: int) -> np.ndarray:
        padded = np.pad(src, ((s, s), (0, 0), (0, 0)), mode="edge")
        vb_im = np.zeros_like(src, dtype=np.float32)

        for i in range(2 * s + 1):
            vb_im += padded[i : i + src.shape[0], :, :]

        vb_im /= 2 * s + 1
        return np.clip(vb_im, 0, 255).astype(np.uint8)

    def smart_blur(self, s: int) -> np.ndarray:
        return self.hblur(self.vblur(self.src, s), s)

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

        return np.clip(dst, 0, 255).astype(dtype=np.uint8)

    def median(self, s: int) -> np.ndarray:

        padded = np.pad(self.src, ((s, s), (s, s), (0, 0)), mode="edge")
        b_im = np.zeros_like(self.src, dtype=np.float32)

        for y in range(self.h):
            for x in range(self.w):
                s_box = padded[y : y + 2 * s + 1, x : x + 2 * s + 1]
                b_im[y, x] = np.median(s_box, axis=(0, 1))

        return np.clip(b_im, 0, 255).astype(dtype=np.uint8)

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
    Image.fromarray(Filter().blur(40))
    Image.fromarray(Filter().smart_blur(40))
