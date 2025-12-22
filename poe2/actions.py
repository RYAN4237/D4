import math
import time
from ctypes import *
import random


class Driver:
    def __init__(self):
        self.vk = {'w': 302, 'a': 401, 's': 402, 'd': 403}
        self.left = 0
        self.top = 0
        self.driver = None
        self.x = 0
        self.y = 0
        # 路径跟踪状态
        self.current_path = None
        self.current_target_index = 0  # 当前目标在路径中的索引
        self.last_target = None  # 上一个目标点，用于判断是否需要重置

    def d_set_ini(self, path_name):
        """
        驱动初始化，成功返回1，失败返回0
        :param path_name: 驱动dll全路径，即使与脚本文件同一个安装路径，也需要写完整路径，比如：X:/XX/XX.dll
        :return:成功返回1，失败返回0
        """
        self.driver = windll.LoadLibrary(path_name)
        st = self.driver.DD_btn(0)
        if st == 1:
            print("驱动初始化成功！")
            self.x = self.left + random.randint(10, 20)
            self.y = self.top + random.randint(10, 20)
            self.driver.DD_mov(self.x, self.y)
            return 1
        else:
            print("驱动初始化失败，请尝试管理员身份运行！")
            return 0
        
    def d_key_press(self, key_name):
        """
        驱动键盘 单击 某键
        :param key_name: 键盘名称，对应键帽上的字符
        :return: 无
        """
        # print(self.vk[key_name])
        self.driver.DD_key(self.vk[key_name], 1)
        time.sleep(0.5)
        self.driver.DD_key(self.vk[key_name], 2)
        time.sleep(0.03)


    def move(self, x, y, path, offset=1, reset_threshold=50):
        """
        根据路径移动角色（带状态记忆，避免原地打转）
        :param x: 当前x坐标
        :param y: 当前y坐标
        :param path: 路径点列表 [(x1, y1), (x2, y2), ...], path[0]通常是起点
        :param offset: 到达目标点的容差范围
        :param reset_threshold: 路径终点距离上次记录超过此值时重置索引
        :return: True表示已到达当前目标点，False表示还在移动中，None表示路径完成或为空
        """
        if not path or len(path) < 1:
            print("路径为空")
            # self.current_path = None
            # self.current_target_index = 0
            # return None

        # 检查是否是新路径或路径终点发生显著变化（重新规划）
        path_end = path[-1]
        if self.current_path is None or self.last_target is None:
            # 初始化路径跟踪
            self.current_path = list(path)
            self.current_target_index = 1 if len(path) > 1 else 0  # 跳过起点
            self.last_target = path_end
            print(f"初始化新路径，共 {len(path)} 个点，起点: {path[0]}, 终点: {path_end}")
        else:
            # 检查终点是否变化太大（说明重新规划了路径）
            dist_to_last_end = abs(path_end[0] - self.last_target[0]) + abs(path_end[1] - self.last_target[1])
            if dist_to_last_end > reset_threshold:
                print(f"检测到路径终点变化 (距离: {dist_to_last_end})，重置路径")
                self.current_path = list(path)
                self.current_target_index = 1 if len(path) > 1 else 0
                self.last_target = path_end

        # 确保索引有效
        if self.current_target_index >= len(path):
            print("已到达路径终点！")
            self.current_path = None
            self.current_target_index = 0
            return None

        # 检查是否已经越过当前目标点及后续点（避免走回头路）
        # 从当前目标开始，找到第一个还未到达的点
        skipped_count = 0
        while self.current_target_index < len(path):
            target_x, target_y = path[self.current_target_index]
            dx = target_x - x
            dy = target_y - y
            distance = abs(dx) + abs(dy)
            
            # 如果这个点已经到达或越过，跳到下一个点
            if distance <= offset:
                skipped_count += 1
                self.current_target_index += 1
                if self.current_target_index >= len(path):
                    print("✓✓ 路径全部完成！")
                    self.current_path = None
                    self.current_target_index = 0
                    return None
            else:
                # 找到第一个还没到达的点
                break
        
        if skipped_count > 0:
            print(f"⚡ 已越过 {skipped_count} 个点，当前目标: [{self.current_target_index}] {path[self.current_target_index]}")

        # 获取当前目标点
        target_x, target_y = path[self.current_target_index]
        dx = target_x - x
        dy = target_y - y
        distance = abs(dx) + abs(dy)

        # 根据距离决定移动方向，优先处理距离更远的轴
        # 添加最小移动阈值，避免微小偏差导致的抖动
        move_threshold = offset * 0.5

        if abs(dx) > abs(dy) and abs(dx) > move_threshold:
            # X轴距离更远，优先处理X方向
            if dx > 0:
                print(f"→ 向右: ({x}, {y}) -> ({target_x}, {target_y}), 距离: {distance}")
                self.d_key_press('d')
            else:
                print(f"← 向左: ({x}, {y}) -> ({target_x}, {target_y}), 距离: {distance}")
                self.d_key_press('a')
        elif abs(dy) > move_threshold:
            # Y轴距离更远或相等，优先处理Y方向
            if dy > 0:
                print(f"↓ 向下: ({x}, {y}) -> ({target_x}, {target_y}), 距离: {distance}")
                self.d_key_press('s')
            else:
                print(f"↑ 向上: ({x}, {y}) -> ({target_x}, {target_y}), 距离: {distance}")
                self.d_key_press('w')
        else:
            # 距离太小，不移动（避免抖动）
            print(f"距离过小 ({distance}), 等待更接近目标")

        return False


    def move_and_click(self, x, y, offset_x=0, offset_y=0):
        x = x + offset_x
        y = y + offset_y
        self.driver.DD_mov(x, y)
        time.sleep(0.05)
        self.driver.DD_btn(4)
        time.sleep(0.03)
        self.driver.DD_btn(8)


if __name__ == "__main__":
    time.sleep(1)
    driver = Driver()
    driver.d_set_ini(r"C:\Repo\D4\poe2\driver.dll")
    while True:
        driver.d_key_press("w")
        time.sleep(1)

