from ctypes import windll

import cv2
import numpy as np
import pyautogui
import win32gui


class CaptureScreen:
    def __init__(self):
        self.win_hwnd = None
        self.win_height = None
        self.win_width = None
        self.win_y = None
        self.win_x = None

    def get_hwnd(self, title):
        """
        获取窗口句柄
        :param title: 窗口标题
        :return: 窗口句柄
        """
        window_list = pyautogui.getWindowsWithTitle(title)
        print(window_list)
        for window in window_list:
            self.win_x, self.win_y, self.win_width, self.win_height, self.win_hwnd = window.left, window.top, window.width, window.height, window._hWnd
        # 窗口置顶
        if self.win_hwnd is not None:
            win32gui.SetForegroundWindow(self.win_hwnd)
        else:
            print("无法获取到正确的窗口句柄")

    def capture(self, x1, y1, x2, y2, x_offset, y_offset):
        """
        截取窗口指定区域的图像
        :param x: 区域左上角x坐标
        :param y: 区域左上角y坐标
        :param width: 区域宽度
        :param height: 区域高度
        :return: 截取的图像数据
        """
        start_x = x1 + x_offset + self.win_x
        start_y = y1 + y_offset + self.win_y
        width = x2 - x1
        height = y2 - y1

        screenshot = pyautogui.screenshot(region=(start_x, start_y, width, height))
        # 将PIL格式图片转换为np数组
        screen_np = np.array(screenshot)
        # 将np数组中图片rbg格式，转换为opencv中bgr格式
        screen_cv = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        return screen_cv

if __name__ == "__main__":
    capturer = CaptureScreen()
    capturer.get_hwnd("Path of Exile 2")
    while True:
        img = capturer.capture(1081, 33, 1261, 176, 1, 1)
        cv2.imshow("Captured Image", img)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()