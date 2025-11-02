import ctypes
import time

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import os
import json


def capture_window(window_title="MapleStory Worlds-Old School Maple"):
    """Capture the game window and return BGR image and hwnd."""
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f"Window not found: {window_title}")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # Try PrintWindow; if not available/fails, fall back to BitBlt copy
    try:
        res = False
        try:
            res = bool(ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2))
        except Exception:
            # silent fallback if PrintWindow symbol isn't available or fails
            res = False
        if not res:
            # fallback: copy from window DC into our bitmap
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
    except Exception as e:
        # ensure resources are released if something goes wrong
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        raise

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
        (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # cleanup
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img_bgr, hwnd


def _load_region_from_json_candidates(candidates):
    """Try load JSON region from a list of candidate paths. Return (region or None, path_used or None)."""
    for p in candidates:
        try:
            if not p:
                continue
            if not os.path.exists(p):
                continue
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            region = data.get('region')
            if (isinstance(region, (list, tuple)) and len(region) == 4
                    and all(isinstance(x, int) for x in region)):
                return region, p
        except Exception:
            continue
    return None, None


def calibrate_minimap_region(window_title="MapleStory Worlds-Old School Maple", json_path: str = None):
    """Return minimap region (left, top, right, bottom).

    Behavior:
    - If a JSON path is provided and valid, use it.
    - Otherwise try common candidate locations (cwd/final/minimap_region.json, module dir/minimap_region.json).
    - Validate loaded coordinates are inside the window rect; otherwise ignore and use defaults.

    The JSON is expected to contain: {"region": [left, top, right, bottom]} with integer coords.
    """
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f"未找到窗口: {window_title}")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    win_w, win_h = right - left, bottom - top

    # Candidate JSON paths to check (order matters)
    candidates = []
    if json_path:
        candidates.append(json_path)
    # common place where the calibrator stores by default (relative to repo root when executed from project)
    candidates.append(os.path.join(os.getcwd(), 'final', 'minimap_region.json'))
    # same directory as this module
    candidates.append(os.path.join(os.path.dirname(__file__), 'minimap_region.json'))

    region, used_path = _load_region_from_json_candidates(candidates)
    if region is not None:
        # validate coords are inside window rect
        l, t, r, b = region
        if 0 <= l < r <= win_w and 0 <= t < b <= win_h:
            print(f"Loaded minimap region from {used_path}: {region}")
            return (l, t, r, b)
        else:
            print(f"Found region in {used_path} but coords {region} are outside window size {win_w}x{win_h}, ignoring")

    # Default region (copied from original)
    minimap_left = 15
    minimap_top = 70
    minimap_right = 210
    minimap_bottom = 200

    return (minimap_left, minimap_top, minimap_right, minimap_bottom)


def save_img(path, img, idx):
    """Save image to path, creating directories as needed.

    `path` should be a directory path where images will be written as "{idx}.jpg".
    The function normalizes to an absolute path and creates the directory if missing.
    """
    # Normalize to absolute path. If user passed a file-like path, treat the dirname as the directory.
    if not path:
        raise ValueError("path must be a non-empty directory path")

    # If path looks like a file (has an extension), use its dirname; otherwise assume it's a directory
    head, tail = os.path.split(path)
    if tail and os.path.splitext(tail)[1]:
        dirpath = head or os.getcwd()
    else:
        dirpath = path

    dirpath = os.path.abspath(dirpath)
    os.makedirs(dirpath, exist_ok=True)
    out_file = os.path.join(dirpath, f"{idx}.jpg")
    # write and return success flag
    success = cv2.imwrite(out_file, img)
    if not success:
        raise IOError(f"Failed to write image to {out_file}")
    return out_file


if __name__ == '__main__':
    # Interactive capture: show window and save on SPACE, quit on 'q' or ESC
    out_dir = r"C:\Repo\D4\final\pic"
    os.makedirs(out_dir, exist_ok=True)

    # pick next available numeric filename so we don't overwrite
    idx = 1
    while os.path.exists(os.path.join(out_dir, f"{idx}.jpg")):
        idx += 1

    win_name = 'capture_preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        print("Press SPACE to save a screenshot to:", out_dir)
        print("Press 'q' or ESC to quit.")
        while True:
            img, hwnd = capture_window()
            # show the BGR image
            cv2.imshow(win_name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                try:
                    saved = save_img(out_dir, img, idx)
                    print(f"Saved {saved}")
                    idx += 1
                except Exception as e:
                    print(f"Failed to save: {e}")
            elif key == ord('q') or key == 27:  # 'q' or ESC
                print('Quitting')
                break
            # small sleep to avoid busy loop; capture_window already takes time
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('Stopped by user')
    finally:
        cv2.destroyAllWindows()
