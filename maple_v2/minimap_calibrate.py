import cv2
import json
import os
import argparse

from final import capture


def parse_args():
    p = argparse.ArgumentParser(description="Interactive minimap calibration tool")
    p.add_argument("--window-title", default="MapleStory Worlds-Old School Maple", help="Window title to capture from")
    p.add_argument("--image", default=None, help="Optional image file to load instead of live capture")
    p.add_argument("--output", default=r"final\minimap_region.json", help="Output JSON file to write region")
    p.add_argument("--max-width", type=int, default=1200)
    p.add_argument("--max-height", type=int, default=800)
    return p.parse_args()


class Calibrator:
    def __init__(self, img, scale, original_shape, out_path):
        self.orig_img = img
        self.scale = scale
        self.out_path = out_path
        self.orig_h, self.orig_w = original_shape[:2]

        if scale != 1.0:
            self.img = cv2.resize(img, (int(self.orig_w * scale), int(self.orig_h * scale)))
        else:
            self.img = img.copy()

        self.clone = self.img.copy()
        self.rect = None  # (x1,y1,x2,y2) in scaled coords
        self.drawing = False
        self.start = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.rect = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x1, y1 = self.start
            self.rect = (x1, y1, x, y)
            self.refresh()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start
            self.rect = (x1, y1, x, y)
            self.refresh()

    def refresh(self):
        self.img = self.clone.copy()
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def show_instructions(self, img):
        text = "Drag to select minimap region. Keys: s=save, r=reset, q=quit"
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        win = "Minimap Calibrator"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.mouse_cb)

        while True:
            display = self.img.copy()
            self.show_instructions(display)
            if self.rect is not None:
                x1, y1, x2, y2 = self.rect
                msg = f"Scaled rect: ({x1},{y1}) - ({x2},{y2})"
                cv2.putText(display, msg, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.rect = None
                self.img = self.clone.copy()
            elif key == ord('s'):
                if self.rect is None:
                    print("No region selected to save")
                    continue
                # convert scaled coords back to original image coords and save
                x1, y1, x2, y2 = self.rect
                sx = 1.0 / self.scale
                ox1 = int(min(x1, x2) * sx)
                oy1 = int(min(y1, y2) * sx)
                ox2 = int(max(x1, x2) * sx)
                oy2 = int(max(y1, y2) * sx)

                region = [int(ox1), int(oy1), int(ox2), int(oy2)]
                data = {
                    'region': region
                }
                # ensure directory exists
                outdir = os.path.dirname(self.out_path)
                if outdir and not os.path.exists(outdir):
                    try:
                        os.makedirs(outdir, exist_ok=True)
                    except Exception as e:
                        print(f"Failed to create directory {outdir}: {e}")

                try:
                    with open(self.out_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved region to {self.out_path}: {region}")
                except Exception as e:
                    print(f"Failed to save file: {e}")

        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        img = cv2.imdecode(__import__('numpy').fromfile(args.image, dtype='uint8'), cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to load image")
            return
    else:
        try:
            img, hwnd = capture.capture_window(args.window_title)
        except Exception as e:
            print(f"Failed to capture window: {e}")
            return

    h, w = img.shape[:2]
    scale_w = min(1.0, args.max_width / float(w))
    scale_h = min(1.0, args.max_height / float(h))
    scale = min(scale_w, scale_h)

    calib = Calibrator(img, scale, img.shape, args.output)
    calib.run()


if __name__ == '__main__':
    main()

