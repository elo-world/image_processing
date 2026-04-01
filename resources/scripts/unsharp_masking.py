if __name__ == "__main__":
    from im_processing import ImProcessing
    from im_processing import auto_timer
else:
    from resources.scripts.im_processing import ImProcessing
    from resources.scripts.im_processing import auto_timer


@auto_timer
class UnsharpMasking(ImProcessing):
    def test():
        pass
