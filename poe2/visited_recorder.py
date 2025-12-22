import os
import cv2



class VisitedRecorder:
    """Record visited points on a big-map canvas and provide overlay/save utilities.

    Usage:
      recorder = VisitedRecorder(big_map_shape=(H,W,3) or (H,W), mask_path=..., radius=4, dedupe_dist=6)
      recorder.mark_point(x,y)
      vis = recorder.overlay_on(big_map)
      recorder.maybe_autosave()
    """

    def __init__(self):
        self.points = set()

    def mark_point(self, x, y):
        self.points.add((x, y))

    def load_point(self):
        return self.points



