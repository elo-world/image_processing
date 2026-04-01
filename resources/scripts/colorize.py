from PIL import Image
import numpy as np

if __name__ == "__main__":
    from im_processing import ImProcessing
    from im_processing import auto_timer
else:
    from resources.scripts.im_processing import ImProcessing
    from resources.scripts.im_processing import auto_timer


@auto_timer
class Colorize(ImProcessing):
    def rgb_to_hsl(self, rgb: np.ndarray) -> np.ndarray:
        r, g, b = rgb / 255

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        l = (cmax + cmin) / 2

        if delta == 0:
            s = 0
            h = 0
        else:
            s = delta / (1 - abs(2 * l - 1))

            if cmax == r:
                h = 60 * (((g - b) / delta) % 6)
            elif cmax == g:
                h = 60 * (((b - r) / delta) + 2)
            else:
                h = 60 * (((r - g) / delta) + 4)

        return np.array([h, s, l])

    def hsl_to_rgb(self, hsl: np.ndarray) -> np.ndarray:
        h, s, l = hsl

        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r = (r + m) * 255
        g = (g + m) * 255
        b = (b + m) * 255

        return np.array([r, g, b])

    def marylyn(self) -> None:
        self.__init__(path="resources/Images/mm.jpg")

        canva = Image.new("RGB", (self.im.width * 4, self.im.height), (255, 255, 255))

        colorized_mm = [self.im]

        hue_values = [0, 180, 90]

        for hue in hue_values:
            colorized_mm.append(Image.fromarray(self.colorize(hue)))

        for x in range(len(colorized_mm)):
            canva.paste(colorized_mm[x], (x * self.im.width, 0))

        return np.array(canva)

    def saturation(self, m: float) -> np.ndarray:
        arr = self.src.copy()

        for y in range(self.h):
            for x in range(self.w):
                h, s, l = self.rgb_to_hsl(self.src[y][x])
                arr[y][x] = self.hsl_to_rgb(np.array([h, s * m, l]))

        return arr.astype(np.uint8)

    def colorize(self, hue: float) -> np.ndarray:
        arr = self.src.copy()

        for y in range(self.h):
            for x in range(self.w):
                _, s, l = self.rgb_to_hsl(self.src[y][x])
                arr[y][x] = self.hsl_to_rgb(np.array([hue % 360, s, l]))

        return arr.astype(np.uint8)


if __name__ == "__main__":
    Image.fromarray(Colorize().marylyn()).show()
