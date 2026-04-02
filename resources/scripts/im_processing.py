from PIL import Image
import numpy as np
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Image processing time ({str(func.__name__).capitalize()}): {end - start:.3f} s.")
        return result

    return wrapper


def auto_timer(cls):
    for name in dir(cls):
        if name.startswith("__"):
            continue
        attr = getattr(cls, name)
        if callable(attr):
            setattr(cls, name, timer(attr))
    return cls


class ImProcessing:
    path: str
    im: Image.Image
    src: np.ndarray
    h: int
    w: int

    def __init__(self, path: str = "resources/images/Lenna.png") -> None:
        self.path = path
        self.im = Image.open(path).convert("RGB")
        self.src = np.array(self.im).astype(dtype=np.float32)
        self.h, self.w, _ = self.src.shape
