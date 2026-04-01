from PIL import Image
import time
import numpy as np

from resources.scripts.filter import Filter
from resources.scripts.colorize import Colorize


class Menu:
    def __init__(self):
        self.runned = True

    def filter_menu(self) -> np.ndarray:
        f_im = Filter("resources/images/Lenna.png")

        with open("filter.txt", "r") as f:
            menu_text = f.read()
            option = input(menu_text).lower()

        start_timer = time.time()

        match option:
            case "b":
                b_level = int(input("Choisir le niveau de flou : "))
                filtered_arr = f_im.blur(b_level)

            case "g":
                g_level = float(input("Choisir l'écart type du bruit (ex: 10) : "))
                filtered_arr = f_im.gaussnoise(g_level)

            case "snp":
                density = float(input("Choisir la densité du bruit : "))
                filtered_arr = f_im.saltnpepper(density)

            case "m":
                m_level = int(input("Choisir le niveau : "))
                filtered_arr = f_im.median(m_level)

            case _:
                filtered_arr = self.filter_menu()

        print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s\n")

        return filtered_arr

    def colorize_menu(self) -> np.ndarray:
        c_im = Colorize("resources/images/Lenna.png")

        with open("colorize.txt", "r") as f:
            menu_text = f.read()
            option = input(menu_text).lower()

        start_timer = time.time()

        match option:
            case "h":
                hue = float(input("Choisir la teinte : "))
                colorized_arr = c_im.colorize(hue)

            case "s":
                s_level = float(input("Saturation multiplié par : "))
                colorized_arr = c_im.saturation(s_level)

            case "mm":
                colorized_arr = Colorize().marylyn()

            case _:
                colorized_arr = self.colorize_menu()

        print(f"Traitement de l'image : {round(time.time() - start_timer, 3)} s\n")

        return colorized_arr

    def menu(self) -> None:
        mode = input("Use filters (f) or colorize (c) an image or quit (q) : ")
        print("\n")
        match mode:
            case "f":
                arr = self.filter_menu()

            case "c":
                arr = self.colorize_menu()

            case "q":
                self.runned = False
                print("See you space cowboy !")

        if self.runned:
            im = Image.fromarray(arr)
            im.show()

    def run(self) -> None:
        while self.runned:
            self.menu()


if __name__ == "__main__":
    Menu().run()
