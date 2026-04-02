import numpy as np
from PIL import ImageEnhance

from resources.scripts.im_processing import ImProcessing


class ASCII(ImProcessing):
    def __init__(
        self, width: int = 70, chars: str = "  ..*%@", contrast: float = 1.5, path: str = "resources/images/Lenna.png"
    ) -> None:
        super().__init__(path)

        gray = self.im.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(contrast)

        aspect_ratio = gray.height / gray.width
        self.width = width
        height = int(aspect_ratio * width * 0.55)
        self.gray = gray.resize((self.width, height))

        self.chars = np.array(list(chars))

    def image_to_ascii(self, save: bool = False) -> str:
        pixels = self.gray.getdata()
        ascii_str = "".join(self.chars[pixel * len(self.chars) // 256] for pixel in pixels)

        lines = [ascii_str[i : i + self.width] for i in range(0, len(ascii_str), self.width)]

        bordered_lines = [f"|   {line}   |" for line in lines]

        border = "_" * (self.width + 8)
        ascii_im = (
            border
            + "\n"
            + f"|   {" "*self.width}   |\n"
            + "\n".join(bordered_lines)
            + "\n"
            + f"|   {" "*self.width}   |\n"
            + border
        )

        if save:
            with open("ascii.txt", "w", encoding="utf-8") as f:
                f.write(ascii_im)

        return ascii_im


if __name__ == "__main__":
    print(ASCII(path="cowboy.png").image_to_ascii(save=True))
