import numpy as np
from PIL import Image

from resources.scripts.im_processing import ImProcessing, auto_timer
from resources.scripts.filter import Filter


@auto_timer
class UnsharpMasking(ImProcessing):
    def unsharp_masking(self, radius: int = 2, amount: float = 1.5) -> np.ndarray:
        blurred = Filter(path=self.path).smart_blur(radius)

        sharpened = self.src + amount * (self.src - blurred)

        return np.clip(sharpened, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    Image.fromarray(UnsharpMasking().unsharp_masking(radius=30)).show()
