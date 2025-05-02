import numpy as np

from consts import H, W


class Drawable:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)

    def world_to_screen(self, zoom):
        center = np.array([W // 2, H // 2])
        return (self.pos - center) * zoom + center

    def draw(self, surf, zoom):
        raise NotImplementedError("Subclasses must implement draw()")


def world_to_screen(pos, zoom):
    center = np.array([W / 2, H / 2])
    return (pos - center) * zoom + center