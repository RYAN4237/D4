import cv2
import numpy as np
import os
from PIL import ImageGrab
import win32gui
import time
import win32con


class MinimapPositionTracker:
    def __init__(self, window_title="MapleStory Worlds-Old School Maple"):
        self.window_title = window_title
        self.minimap_region = None  # 小地图区域
        self.target_positions = []  # 目标位置列表
        self.current_minimap_pos = None
        self.position_tolerance = 5  # 小地图上的像素容忍度

    def calibrate_minimap_region(self):
        """校准小地图区域（首次运行时调用）"""
        hwnd = win32gui.FindWindow(None, self.window_title)
        if not hwnd:
            raise Exception("未找到游戏窗口")

        # 激活窗口
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)

        # 获取窗口坐标
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)

        # 根据你的截图，小地图大致在左上角
        # 这个区域需要根据实际情况微调
        minimap_left = left + 15
        minimap_top = top + 65
        minimap_right = left + 280
        minimap_bottom = top + 200

        self.minimap_region = (
        minimap_left, minimap_top, minimap_right, minimap_bottom)
        print(f"小地图区域设置为: {self.minimap_region}")

        # 截取小地图进行验证
        minimap_img = ImageGrab.grab(bbox=self.minimap_region)
        minimap_cv = cv2.cvtColor(np.array(minimap_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite('minimap_calibration.png', minimap_cv)
        print("已保存小地图校准图片: minimap_calibration.png")

    def find_yellow_dot(self, minimap_img):
        """在小地图中找到黄色点（玩家位置）"""
        # 转换为HSV色彩空间，更好地检测黄色
        hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)

        # 黄色的HSV范围（需要根据实际情况调整）
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # 创建黄色mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大的轮廓（假设是玩家点）
            largest_contour = max(contours, key=cv2.contourArea)

            # 计算轮廓中心
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        return None

    def get_current_minimap_position(self):
        """获取当前在小地图上的位置"""
        if not self.minimap_region:
            self.calibrate_minimap_region()

        try:
            # 截取小地图
            minimap_img = ImageGrab.grab(bbox=self.minimap_region)
            minimap_cv = cv2.cvtColor(np.array(minimap_img), cv2.COLOR_RGB2BGR)

            # 找到黄色点
            yellow_dot_pos = self.find_yellow_dot(minimap_cv)

            if yellow_dot_pos:
                self.current_minimap_pos = yellow_dot_pos
                return yellow_dot_pos
            else:
                print("未找到黄色标记点")
                return None

        except Exception as e:
            print(f"获取小地图位置失败: {e}")
            return None

    def add_target_position(self, minimap_x, minimap_y, action_name,
                            action_func):
        """添加目标位置（小地图坐标）"""
        self.target_positions.append({
            'x': minimap_x,
            'y': minimap_y,
            'action_name': action_name,
            'action_func': action_func,
            'executed': False
        })
        print(f"已添加目标位置: ({minimap_x}, {minimap_y}) - {action_name}")

    def check_and_execute_actions(self):
        """检查是否到达目标位置并执行动作"""
        pos = self.get_current_minimap_position()
        if not pos:
            return

        for target in self.target_positions:
            # 计算在小地图上的距离
            distance = np.sqrt(
                (pos[0] - target['x']) ** 2 + (pos[1] - target['y']) ** 2)

            if distance <= self.position_tolerance:
                if not target['executed']:
                    print(
                        f"到达目标位置 ({target['x']}, {target['y']})，执行动作: {target['action_name']}")
                    target['action_func']()
                    target['executed'] = True
            else:
                target['executed'] = False

    def debug_minimap_position(self):
        """调试模式：显示当前位置并保存标记图片"""
        pos = self.get_current_minimap_position()
        if pos:
            # 截取小地图并标记位置
            minimap_img = ImageGrab.grab(bbox=self.minimap_region)
            minimap_cv = cv2.cvtColor(np.array(minimap_img), cv2.COLOR_RGB2BGR)

            # 在检测到的位置画圈
            cv2.circle(minimap_cv, pos, 3, (0, 0, 255), 2)  # 红圈标记
            cv2.putText(minimap_cv, f"Player: {pos}", (pos[0] + 5, pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 标记所有目标位置
            for i, target in enumerate(self.target_positions):
                cv2.circle(minimap_cv, (target['x'], target['y']), 5,
                           (0, 255, 0), 2)
                cv2.putText(minimap_cv, f"T{i + 1}",
                            (target['x'] + 5, target['y'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imwrite('minimap_debug.png', minimap_cv)
            print(f"当前小地图位置: {pos}，调试图片已保存: minimap_debug.png")
            return pos
        else:
            print("未检测到玩家位置")
            return None

    def run_monitoring(self, interval=1.0, debug=False):
        """持续监控位置"""
        print("开始小地图位置监控...")
        try:
            while True:
                if debug:
                    self.debug_minimap_position()
                else:
                    self.check_and_execute_actions()
                    if self.current_minimap_pos:
                        print(f"小地图位置: {self.current_minimap_pos}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("监控已停止")


# 示例动作函数
def action_at_position_1():
    print("执行位置1的动作：拾取物品!")
    # import pyautogui
    # pyautogui.press('z')


def action_at_position_2():
    print("执行位置2的动作：攻击怪物!")
    # import pyautogui
    # pyautogui.press('ctrl')


def action_at_position_3():
    print("执行位置3的动作：使用技能!")
    # import pyautogui
    # pyautogui.press('a')


# 使用示例
if __name__ == "__main__":
    tracker = MinimapPositionTracker()

    # 首先校准小地图区域
    tracker.calibrate_minimap_region()

    # 调试模式：查看当前位置检测效果
    print("=== 调试模式：检测当前位置 ===")
    current_pos = tracker.debug_minimap_position()

    if current_pos:
        print(f"检测到位置: {current_pos}")

        # 添加目标位置（需要根据实际游戏情况设置坐标）
        # 这些坐标是小地图上的像素坐标
        tracker.add_target_position(100, 80, "位置1动作", action_at_position_1)
        tracker.add_target_position(120, 100, "位置2动作", action_at_position_2)
        tracker.add_target_position(80, 120, "位置3动作", action_at_position_3)

        # 开始监控
        choice = input("是否开始位置监控？(y/n): ")
        if choice.lower() == 'y':
            tracker.run_monitoring(interval=0.5, debug=True)  # debug=True显示调试信息
    else:
        print("无法检测到玩家位置，请检查小地图区域设置或黄色检测参数")