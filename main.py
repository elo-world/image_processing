from PIL import Image
import time
import numpy as np
import sys

from resources.scripts.ascii import ASCII
from resources.scripts.filter import Filter
from resources.scripts.colorize import Colorize
from resources.scripts.convolution_kernels import ConvolutionKernels
from resources.scripts.kernels import KERNELS


class Menu:
    def __init__(self):
        self.runned = True

    def filter_menu(self, path: str = "resources/images/Lenna.png") -> np.ndarray:
        f_im = Filter(path=path)

        with open("resources/filter.txt", "r") as f:
            menu_text = f.read()
            option = input(menu_text).lower()

        start_timer = time.time()

        match option:
            case "b":
                b_level = int(input("Select the blur level (ex: 3): "))
                filtered_arr = f_im.blur(b_level)

            case "g":
                g_level = float(input("Select the standard deviation of the noise (ex: 10): "))
                filtered_arr = f_im.gaussnoise(g_level)

            case "snp":
                density = float(input("Select the noise density (ex: 0.1): "))
                filtered_arr = f_im.saltnpepper(density)

            case "m":
                m_level = int(input("Choose the level (ex: 3): "))
                filtered_arr = f_im.median(m_level)

            case _:
                print("\n")
                filtered_arr = self.filter_menu()

        print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s\n")

        return filtered_arr

    def colorize_menu(self, path: str = "resources/images/Lenna.png") -> np.ndarray:
        c_im = Colorize(path=path)

        with open("resources/colorize.txt", "r") as f:
            menu_text = f.read()
            option = input(menu_text).lower()

        match option:
            case "h":
                hue = float(input("Choose a shade (ex: 70): "))
                colorized_arr = c_im.colorize(hue)

            case "s":
                s_level = float(input("Saturation multiplied by (ex: 1.2): "))
                colorized_arr = c_im.saturation(s_level)

            case "mm":  # EasterEgg
                colorized_arr = Colorize().marylyn()

            case _:
                print("\n")
                colorized_arr = self.colorize_menu()
        start_timer = time.time()
        print(f"Image processing time: {round(time.time() - start_timer, 3)} s\n")

        return colorized_arr

    def convolution_kernels_menu(self, path: str = "resources/images/Lenna.png") -> np.ndarray:
        k_im = ConvolutionKernels(path=path)

        with open("resources/convolution_kernels.txt", "r") as f:
            menu_text = f.read()
            option = input(menu_text).lower()

        start_timer = time.time()

        match option:
            case "k":
                kernels = {kernel[0] + kernel[-1]: kernel for kernel in KERNELS}
                key = ""
                while not key in kernels:
                    key = input(
                        f"Choose a kernel:\n{"\n".join([f"- {key}: {kernel.capitalize()}" for key, kernel in kernels.items()])}\nEnter an option: "
                    ).lower()
                kernels_arr = k_im.apply_kernel(kernel=kernels[key])

            case "s":
                modes = {"g": "gray", "ig": "invert_gray", "c": "color", "o": "overlay"}
                key = input(
                    f"Choose a mode:\n{"\n".join([f"- {key}: {" ".join(mode.split("_")).capitalize()}" for key, mode in modes.items()])}\nEnter an option: "
                )
                kernels_arr = k_im.sobel(mode=modes[key])

            case _:
                print("\n")
                kernels_arr = self.convolution_kernels_menu()

        print(f"Image processing time: {round(time.time() - start_timer, 3)} s\n")

        return kernels_arr

    def menu(self, path: str = "resources/images/Lenna.png") -> None:
        mode = input("\nUse filters (f), colorize (c) or convolution kernels (k) for your image or exit (e) : ")
        print("\n")
        match mode:
            case "f":
                arr = self.filter_menu(path)

            case "c":
                arr = self.colorize_menu(path)

            case "k":
                arr = self.convolution_kernels_menu(path)

            case "e":
                self.runned = False
                print("SEE YOU SPACE COWBOY...\n")

            case _:
                self.runned = False
                print(f"There is no mode {mode}.")

        if self.runned:
            im = Image.fromarray(arr)
            im.show()

    def run(self) -> None:
        path = sys.argv[1] if len(sys.argv) > 1 else input("Path of your image : ")
        if path == "default" or path == "d":
            path = "resources/images/Lenna.png"

        print(ASCII(path=path).image_to_ascii())

        with open("resources/header.txt", "r") as f:
            print(f.read())

        while self.runned:
            self.menu(path)


if __name__ == "__main__":
    Menu().run()
