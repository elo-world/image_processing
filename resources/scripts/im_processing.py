from PIL import Image
import numpy as np


class ImProcessing:
    def __init__(self, path: str = "resources/images/mm.jpg") -> None:
        self.im = Image.open(path)
        self.src = np.array(self.im)
        self.h, self.w, _ = self.src.shape
