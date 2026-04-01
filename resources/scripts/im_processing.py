from PIL import Image
import numpy as np
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"The {str(func.__name__).capitalize()} function took {end - start:.3f} seconds to execute.")
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
    def __init__(self, path: str = "resources/images/Lenna.png") -> None:
        self.im = Image.open(path)
        self.src = np.array(self.im)
        self.h, self.w, _ = self.src.shape
