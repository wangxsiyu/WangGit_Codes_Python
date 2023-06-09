import os

class W_io():
    def W_mkdir(folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder

