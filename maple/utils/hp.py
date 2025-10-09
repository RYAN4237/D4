import numpy as np
import cv2

def extract_status_bars_precise(img):
    """精确提取HP和MP条区域"""
    h, w = img.shape[:2]

    bar_region_y1 = h - 50
    bar_region_y2 = h - 10
    bar_region_x1 = 320
    bar_region_x2 = 800

    status_region = img[bar_region_y1:bar_region_y2, bar_region_x1:bar_region_x2]

    hp_region = status_region[:, 100:300]
    mp_region = status_region[:, 300:500]

    return hp_region, mp_region, status_region, (bar_region_x1, bar_region_y1)


def calculate_bar_fill_ratio(bar_region, color_type='red'):
    """通过检测条的填充长度计算百分比"""
    h, w = bar_region.shape[:2]
    b, g, r = cv2.split(bar_region)

    if color_type == 'red':
        mask1 = cv2.inRange(r, 150, 255)
        mask2 = cv2.inRange(g, 0, 150)
        mask3 = cv2.inRange(b, 0, 150)

        r_minus_g = cv2.subtract(r, g)
        r_minus_b = cv2.subtract(r, b)

        mask4 = cv2.inRange(r_minus_g, 50, 255)
        mask5 = cv2.inRange(r_minus_b, 80, 255)

        mask = cv2.bitwise_and(mask1, mask2)
        mask = cv2.bitwise_and(mask, mask3)
        mask = cv2.bitwise_and(mask, mask4)
        mask = cv2.bitwise_and(mask, mask5)

    else:
        mask1 = cv2.inRange(b, 150, 255)
        mask2 = cv2.inRange(r, 0, 150)
        mask3 = cv2.inRange(g, 0, 150)

        b_minus_r = cv2.subtract(b, r)
        b_minus_g = cv2.subtract(b, g)

        mask4 = cv2.inRange(b_minus_r, 80, 255)
        mask5 = cv2.inRange(b_minus_g, 50, 255)

        mask = cv2.bitwise_and(mask1, mask2)
        mask = cv2.bitwise_and(mask, mask3)
        mask = cv2.bitwise_and(mask, mask4)
        mask = cv2.bitwise_and(mask, mask5)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    col_sum = np.sum(mask, axis=0)
    threshold = h * 5
    active_cols = np.where(col_sum > threshold)[0]

    if len(active_cols) == 0:
        return 0.0, mask, 0, 0

    bar_start = np.min(active_cols)
    bar_end = np.max(active_cols)
    bar_width = bar_end - bar_start + 1
    full_bar_width = w

    percentage = (bar_width / full_bar_width) * 100
    percentage = max(0, min(100, percentage))

    return percentage, mask, bar_start, bar_end