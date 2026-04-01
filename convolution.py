from im_processing import ImProcessing
from PIL import Image
import numpy as np

from kernels import KERNELS

class Convolution(ImProcessing):
    def __init__(self, path):
        super().__init__(path)

    def sobel(self, mode="gray"):
        s = 1
        h, w, _ = self.src.shape

        if mode == "gray":
            gray = np.mean(self.src, axis=2)
            padded = np.pad(gray, ((s, s), (s, s)), mode='reflect')

            G = np.zeros((h, w))

            for y in range(h):
                for x in range(w):
                    region = padded[y:y+3, x:x+3]
                    gx = np.sum(region * KERNELS["sobel_x"])
                    gy = np.sum(region * KERNELS["sobel_y"])
                    G[y, x] = np.sqrt(gx**2 + gy**2)
            
            G = G / G.max() * 255

            return G.astype(np.uint8)
        
        elif mode == "invert_gray":
            gray = np.mean(self.src, axis=2)
            padded = np.pad(gray, ((s, s), (s, s)), mode='reflect')

            G = np.zeros((h, w))

            for y in range(h):
                for x in range(w):
                    region = padded[y:y+3, x:x+3]
                    gx = np.sum(region * KERNELS["sobel_x"])
                    gy = np.sum(region * KERNELS["sobel_y"])
                    G[y, x] = np.sqrt(gx**2 + gy**2)

            G = G / G.max() * 255

            G = np.array(255) - G

            return G.astype(np.uint8)

        elif mode == "color":
            padded = np.pad(self.src, ((s, s), (s, s), (0, 0)), mode='reflect')
            result = np.zeros_like(self.src, dtype=float)

            for y in range(h):
                for x in range(w):
                    region = padded[y:y+3, x:x+3]

                    gx = np.sum(region * KERNELS["sobel_x"].reshape(3,3,1), axis=(0,1))
                    gy = np.sum(region * KERNELS["sobel_y"].reshape(3,3,1), axis=(0,1))

                    result[y, x] = np.sqrt(gx**2 + gy**2)

            return np.clip(result, 0, 255).astype(np.uint8)

        elif mode == "overlay":
            gray = np.mean(self.src, axis=2)
            padded = np.pad(gray, ((s, s), (s, s)), mode='reflect')

            G = np.zeros((h, w))

            for y in range(h):
                for x in range(w):
                    region = padded[y:y+3, x:x+3]
                    gx = np.sum(region * KERNELS["sobel_x"])
                    gy = np.sum(region * KERNELS["sobel_y"])
                    G[y, x] = np.sqrt(gx**2 + gy**2)

            G = G / G.max()

            result = self.src * G.reshape(h, w, 1)

            return np.clip(result, 0, 255).astype(np.uint8)

        else:
            raise ValueError("mode must be 'gray', 'invert_gray', 'color', or 'overlay'")

    def apply_kernel(self, kernel):
        k = KERNELS[kernel]
        s = k.shape[0] // 2

        padded = np.pad(self.src, ((s, s), (s, s), (0, 0)), mode='edge')
        b_im = self.src.copy()

        for y in range(self.h):
            for x in range(self.w):
                s_box = padded[y : y + 2*s + 1, x : x + 2*s + 1]
                value = np.sum(s_box * k.reshape(*k.shape, 1), axis=(0, 1))
                b_im[y, x] = np.clip(value, 0, 255)

        return b_im.astype(np.uint8)


c = Convolution("resources/images/mm.jpg").sobel("overlay")
Image.fromarray(c).show()
