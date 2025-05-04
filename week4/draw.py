import numpy as np

from consts import H, W

class Drawable:
    zoom = 1.0  # Default zoom level

    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)

    def world_to_screen(self):
        center = np.array([W // 2, H // 2])
        return (self.pos - center) * Drawable.zoom + center

    def draw(self, surf):
        raise NotImplementedError("Subclasses must implement draw()")

    @classmethod
    def set_zoom(cls, z):
        cls.zoom = z