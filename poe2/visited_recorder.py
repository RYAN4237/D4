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

    def __init__(self, big_map, mask_path, radius=4):
        # big_map_shape can be (H,W,3) or (H,W)
        self.big_map = big_map
        self.mask_path = mask_path
        self.radius = radius
        self.points = []

    def mark_point(self, x, y):
        """Mark a player point (big-map pixel coords). Applies dedupe and writes a filled circle to mask.
        Returns True if the point was recorded, False if skipped due to dedupe.
        """
        if os.path.exists(self.mask_path):
            img = cv2.imread(self.mask_path)
        else:
            img = self.big_map
        cv2.circle(img, (x, y), self.radius, 255, -1)
        cv2.imwrite(self.mask_path, img)
        self.points.append((x, y))

    def load_point(self):
        return self.points



