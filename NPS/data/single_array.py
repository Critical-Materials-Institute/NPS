__author__ = 'Fei Zhou'

from .longclip import longclip

class single_image(longclip):
    def __len__(self):
        return len(self.flat)

    def __getitem__(self, i):
        return self.flat[i]
