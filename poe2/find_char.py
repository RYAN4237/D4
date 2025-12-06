"""
简单的 cv2.inRange 检测示例
- 把用户提供的 HSV (H=度 0-360, S/V=百分比 0-100) 列表转换为 OpenCV HSV (H:0-179, S/V:0-255)
- 为每个颜色生成 lower/upper（处理 Hue 环绕）
- 合并所有颜色的 mask，形态学去噪，连通域提取质心并标注图片

用法:
    python simple_inrange_detection.py path/to/map.png

输出:
    - 标注图片 saved as annotated_map.png
    - 控制台打印检测到的点 (像素坐标 & 归一化百分比 & 面积)
"""
import sys
import cv2
import numpy as np
from typing import List, Tuple

def degpct_to_opencv(h_deg: float, s_pct: float, v_pct: float) -> Tuple[int,int,int]:
    """Convert H (0..360 deg), S% (0..100), V% (0..100) to OpenCV HSV scale (H:0..179, S:0..255, V:0..255)."""
    h = int(round(h_deg * 179.0 / 360.0)) % 180
    s = int(round(s_pct * 255.0 / 100.0))
    v = int(round(v_pct * 255.0 / 100.0))
    return h, s, v

def hsv_range_opencv(h_op: int, s_op: int, v_op: int,
                     h_tol_deg: float, s_tol_pct: float, v_tol_pct: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (lower, upper) np.uint8 ranges for cv2.inRange.
    Handles hue wrap-around by returning 1 or 2 ranges.
    h_tol_deg / s_tol_pct / v_tol_pct are in human units (deg / %), converted inside.
    """
    h_tol = int(round(h_tol_deg * 179.0 / 360.0))
    s_tol = int(round(s_tol_pct * 255.0 / 100.0))
    v_tol = int(round(v_tol_pct * 255.0 / 100.0))

    low_h = h_op - h_tol
    high_h = h_op + h_tol
    low_s = max(0, s_op - s_tol)
    high_s = min(255, s_op + s_tol)
    low_v = max(0, v_op - v_tol)
    high_v = min(255, v_op + v_tol)

    ranges = []
    if low_h >= 0 and high_h <= 179:
        ranges.append((np.array([low_h, low_s, low_v], dtype=np.uint8),
                       np.array([high_h, high_s, high_v], dtype=np.uint8)))
    else:
        # wrap-around
        if low_h < 0:
            # [0, high_h] and [179+low_h, 179]
            ranges.append((np.array([0, low_s, low_v], dtype=np.uint8),
                           np.array([high_h, high_s, high_v], dtype=np.uint8)))
            ranges.append((np.array([179 + low_h, low_s, low_v], dtype=np.uint8),
                           np.array([179, high_s, high_v], dtype=np.uint8)))
        else:
            # high_h > 179: [low_h,179] and [0, high_h-180]
            ranges.append((np.array([low_h, low_s, low_v], dtype=np.uint8),
                           np.array([179, high_s, high_v], dtype=np.uint8)))
            ranges.append((np.array([0, low_s, low_v], dtype=np.uint8),
                           np.array([high_h - 180, high_s, high_v], dtype=np.uint8)))
    return ranges

def detect_and_annotate(img: np.ndarray,
                        hsv_list: List[Tuple[float,float,float]],
                        h_tol_deg: float = 12.0,
                        s_tol_pct: float = 25.0,
                        v_tol_pct: float = 30.0,
                        area_min: float = 3.0,
                        morph_kernel: int = 5,
                        save_path: str = "annotated_map.png"):
    """
    Simple pipeline:
      - read image -> convert to HSV
      - for each hsv in hsv_list compute ranges and build mask
      - combine masks, morphology, connectedComponentsWithStats
      - annotate and save image, print detections
    """

    h_img, w_img = img.shape[:2]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Build mask for all provided colors
    final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for (h_deg, s_pct, v_pct) in hsv_list:
        h_op, s_op, v_op = degpct_to_opencv(h_deg, s_pct, v_pct)
        ranges = hsv_range_opencv(h_op, s_op, v_op, h_tol_deg, s_tol_pct, v_tol_pct)
        mask_color = None
        for low, high in ranges:
            m = cv2.inRange(hsv_img, low, high)
            if mask_color is None:
                mask_color = m
            else:
                mask_color = cv2.bitwise_or(mask_color, m)
        if mask_color is not None:
            final_mask = cv2.bitwise_or(final_mask, mask_color)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_min:
            continue
        M = cv2.moments(contour)

        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            return cx, cy

    return 0, 0


if __name__ == "__main__":
    # 用户给的颜色组合
    hsv_inputs = [
        (25.0, 85.0, 69.0)    ]

    img_path = sys.argv[1] if len(sys.argv) > 1 else "img.png"
    detect_and_annotate(
        image_path=img_path,
        hsv_list=hsv_inputs,
        h_tol_deg=12.0,     # 色相容差(度) - 可根据需要减小或增大
        s_tol_pct=25.0,     # 饱和度容差(%) - 可调整
        v_tol_pct=30.0,     # 亮度容差(%) - 可调整，第三组亮度较低可适当减小以避免误检
    )