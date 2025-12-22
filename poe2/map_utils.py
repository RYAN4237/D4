import cv2
import numpy as np


class MapUtils:
    @staticmethod
    def findpic(big_pic, mini_pic, threshold, method=0):
        if method == 0:
            template_res = cv2.matchTemplate(big_pic, mini_pic, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_res)
            if max_val >= threshold:
                return max_loc
            else:
                print(f"[TM_CCORR_NORMED] match failed: max_val={max_val} < threshold={threshold}")
                return 0, 0  # match failed

        elif method == 1:
            template_res = cv2.matchTemplate(big_pic, mini_pic, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_res)
            if max_val >= threshold:
                return max_loc
            else:
                print(f"[TM_CCOEFF_NORMED] match failed: max_val={max_val} < threshold={threshold}")
                return 0, 0  # match failed
        elif method == 2:
            template_res = cv2.matchTemplate(big_pic, mini_pic, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_res)
            if min_val <= threshold:
                return min_loc
            else:
                print(f"[TM_SQDIFF_NORMED] match failed: min_val={min_val} > threshold={threshold}")
                return 0, 0


    @staticmethod
    def merge_blue_into_binary(img_bgr, blue_lower=(90, 120, 80), blue_upper=(110, 255, 255),
                               dilate_before_close=True, kernel_size=(3, 3), blue_value=127):
        """
        把 blue_mask 合并到二值图 bin_gray（来自 adaptiveThreshold），并把 blue 区设为 blue_value。
        参数:
          - img_bgr: 原始 BGR 图
          - bin_gray: adaptiveThreshold 的结果 (单通道 0/255)
          - blue_lower/blue_upper: HSV 范围（建议基于样本 [100,188,178] 使用 H~100, S 下限 <=188）
          - dilate_before_close: 是否先 dilate 再 close（避免 open 吃掉细线）
          - kernel_size: 形态学核尺寸
          - blue_value: 合并后蓝色像素值 (127)
        返回:
          - three_map: 单通道 uint8，值为 {0, blue_value, 255}
          - vis: BGR 可视化图（blue->蓝色, obstacle->白, free->黑）
        """
        big_map_mask_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 你的原 pipeline
        big_map_mask = cv2.inRange(big_map_mask_hsv,
                                   np.array([20, 40, 150], dtype=np.uint8),
                                   np.array([130, 190, 255], dtype=np.uint8))
        big_map_new = cv2.bitwise_and(img_bgr, img_bgr, mask=big_map_mask)
        big_map_new_gray = cv2.cvtColor(big_map_new, cv2.COLOR_BGR2GRAY)
        big_map_new_gray = cv2.adaptiveThreshold(big_map_new_gray, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 5, 1)
        big_map_new_gray = cv2.medianBlur(big_map_new_gray, 3)
        big_map_new_gray = cv2.dilate(big_map_new_gray, (3, 3), iterations=1)
        big_map_new_gray = cv2.morphologyEx(big_map_new_gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        big_map_new_gray = cv2.erode(big_map_new_gray, (3, 3), iterations=1)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(blue_lower, dtype=np.uint8)
        upper = np.array(blue_upper, dtype=np.uint8)
        blue_mask = cv2.inRange(hsv, lower, upper)

        k = np.ones(kernel_size, np.uint8)

        # 推荐先把线变粗以保持连通性：dilate -> close -> erode
        # if dilate_before_close:
        #     blue_mask = cv2.dilate(blue_mask, k, iterations=1)
        # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k, iterations=1)
        # if dilate_before_close:
        #     blue_mask = cv2.erode(blue_mask, k, iterations=1)

        # 合并：把 adaptiveThreshold 的结果作为基础，然后覆盖 blue 区为 127
        three_map = big_map_new_gray.copy().astype(np.uint8)
        three_map[blue_mask > 0] = blue_value

        return three_map

    @staticmethod
    def click_for_color():
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(param[y, x])

        cv2.namedWindow("win")
        img = cv2.imread("current_map_hsv.png")
        img = cv2.resize(img, (img.shape[1]*6, img.shape[0]*6), interpolation=cv2.INTER_NEAREST)
        cv2.setMouseCallback("win", on_mouse, img)
        cv2.imshow("win", img)

        # compare
        big_map_mask = cv2.inRange(img,
                                   np.array([20, 40, 150], dtype=np.uint8),
                                   np.array([130, 190, 255], dtype=np.uint8))
        cv2.imshow("mask", big_map_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    MapUtils.click_for_color()