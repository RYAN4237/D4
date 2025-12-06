"""
智能小地图拼接 - 带实时监控版本
只在检测到真实移动时才拼接，避免累积误差
"""
import time
from capture import CaptureScreen
import numpy as np
import cv2

class SmartMinimapStitcher:
    def __init__(self, x1, y1, x2, y2, name="MapleStory Worlds-Old School Maple"):
        self.jy = CaptureScreen()
        self.jy.get_hwnd(name)
        # time.sleep(1.5)

        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

        # 创建画布
        self.canvas_size = 3000
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 200

        # 初始化位置（画布中心）
        self.canvas_x = self.canvas_size // 2
        self.canvas_y = self.canvas_size // 2

        # 统计信息
        self.frame_count = 0
        self.stitch_count = 0
        self.last_minimap = None

        print("="*60)
        print("智能小地图拼接工具")
        print("="*60)
        print(f"截图区域: ({x1},{y1}) -> ({x2},{y2})")
        print(f"画布大小: {self.canvas_size} x {self.canvas_size}")

    def capture_minimap(self):
        """截取小地图"""
        return self.jy.capture(self.x1, self.y1, self.x2, self.y2, 1, 1)

    def detect_movement(self, img1, img2, threshold=3.0):
        """
        检测两帧之间是否有真实移动（优化版：去除边框后更准确）
        返回: (has_moved, dx, dy, confidence)
        """
        # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        # Enhance mini_map to reduce noise
        img1_mask = cv2.inRange(img1_hsv,
                                    np.array([20, 60, 150], dtype=np.uint8),
                                    np.array([130, 190, 220], dtype=np.uint8))
        img1_new = cv2.bitwise_and(img1, img1, mask=img1_mask)
        gray1 = cv2.cvtColor(img1_new, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 5, 1)
        gray1 = cv2.medianBlur(gray1, 3)


        # Enhance big_map to reduce noise
        img2_mask = cv2.inRange(img2_hsv,
                                   np.array([20, 60, 150], dtype=np.uint8),
                                   np.array([130, 190, 220], dtype=np.uint8))
        img2_new = cv2.bitwise_and(img2, img2, mask=img2_mask)
        gray2 = cv2.cvtColor(img2_new, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 5, 1)
        gray2 = cv2.medianBlur(gray2, 3)
        # ------------------------------------------

        # 计算帧差
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = np.mean(diff)

        # 提高阈值，因为去掉边框后应该更稳定
        if mean_diff < threshold:
            return False, 0, 0, 0

        # 使用相位相关检测精确位移
        try:
            shift, response = cv2.phaseCorrelate(
                np.float32(gray1),
                np.float32(gray2)
            )

            dx, dy = int(round(shift[0])), int(round(shift[1]))

            # 如果位移太小，认为没有移动
            if abs(dx) < 1 and abs(dy) < 1:
                return False, 0, 0, response

            # 提高置信度要求（去掉边框后应该更准确）
            if response < 0.25:
                return False, 0, 0, response

            # 如果位移太大，可能是误判
            if abs(dx) > 50 or abs(dy) > 50:
                return False, 0, 0, response

            return True, dx, dy, response

        except Exception as e:
            print(f"  警告: 位移检测失败 - {e}")
            return False, 0, 0, 0

    def stitch(self, minimap):
        """将小地图拼接到画布上（完全覆盖模式，避免色差）"""
        h, w = minimap.shape[:2]

        # 计算在画布上的位置
        y1 = self.canvas_y
        y2 = y1 + h
        x1 = self.canvas_x
        x2 = x1 + w

        # 边界检查
        if y1 < 0 or x1 < 0 or y2 > self.canvas_size or x2 > self.canvas_size:
            print(f"  ⚠️ 警告: 超出画布范围 ({x1},{y1})-({x2},{y2})")
            return False

        # 完全覆盖模式：直接复制整个小地图到画布
        # 这样可以避免颜色混合导致的色差问题
        self.canvas[y1:y2, x1:x2] = minimap

        return True



    def run(self):
        """主循环"""
        # 截取初始帧
        minimap = self.capture_minimap()
        h, w = minimap.shape[:2]

        print(f"\n小地图尺寸: {w} x {h}")
        print(f"初始位置: ({self.canvas_x}, {self.canvas_y})")

        # 放置初始帧到画布中心
        self.canvas[self.canvas_y:self.canvas_y+h, self.canvas_x:self.canvas_x+w] = minimap
        self.last_minimap = minimap.copy()

        print("\n开始监控...")
        print("请在游戏中移动角色！")
        print("-"*60)
        print("快捷键:")
        print("  q - 退出并保存")
        print("  s - 立即保存当前画布")
        print("  r - 重置画布")
        print("  空格 - 暂停/继续")
        print("-"*60)

        cv2.namedWindow("minimap_stitch_smart", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('minimap_stitch_smart', 800, 800)

        paused = False
        last_report_time = time.time()

        try:
            while True:
                if not paused:
                    # 截取新帧
                    minimap_new = self.capture_minimap()
                    self.frame_count += 1

                    # 检测移动
                    has_moved, dx, dy, confidence = self.detect_movement(
                        self.last_minimap,
                        minimap_new
                    )


                    if has_moved:
                        # 更新画布位置
                        self.canvas_x -= dx
                        self.canvas_y -= dy

                        # 拼接
                        success = self.stitch(minimap_new)

                        if success:
                            self.stitch_count += 1
                            print(f"[{self.stitch_count:03d}] 移动: ({dx:+3d}, {dy:+3d}), "
                                  f"位置: ({self.canvas_x}, {self.canvas_y}), "
                                  f"置信度: {confidence:.3f}")

                        # 更新上一帧
                        self.last_minimap = minimap_new.copy()

                    # 每5秒报告一次状态
                    current_time = time.time()
                    if current_time - last_report_time > 5:
                        print(f"  [状态] 总帧数: {self.frame_count}, "
                              f"拼接次数: {self.stitch_count}, "
                              f"拼接率: {self.stitch_count/self.frame_count*100:.1f}%")
                        last_report_time = current_time

                # 显示画布（裁剪到有内容的区域）
                display_canvas = self.canvas.copy()

                # 在当前位置画一个红框表示当前视野
                cv2.rectangle(display_canvas,
                            (self.canvas_x, self.canvas_y),
                            (self.canvas_x + w, self.canvas_y + h),
                            (0, 0, 255), 2)

                # cv2.imshow('minimap_stitch_smart', display_canvas)

                # 处理按键
                key = cv2.waitKey(50) & 0xFF

                # 降低CPU占用
                time.sleep(0.05)

                # 保存最终结果
                final_filename = 'final_smart_stitch.png'
                cv2.imwrite(final_filename, self.canvas)

                # 裁剪掉多余的灰色背景
                gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                mask = gray_canvas != 200
                coords = np.argwhere(mask)

                if len(coords) > 0:
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0)
                    cropped = self.canvas[y0:y1 + 1, x0:x1 + 1]
                    cropped_filename = 'final_smart_stitch_cropped.png'
                    cv2.imwrite(cropped_filename, cropped)

        except KeyboardInterrupt:
            print("\n中断退出")



        print(f"✓ 已保存完整版: {final_filename}")
        print(f"\n统计信息:")
        print(f"  总帧数: {self.frame_count}")
        print(f"  拼接次数: {self.stitch_count}")
        print(f"  拼接率: {self.stitch_count/max(1, self.frame_count)*100:.1f}%")

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 小地图区域（去掉边框后的纯地图内容）
    # 原始区域: (10,60)-(245,156) 包含边框
    # 优化后: 去掉约8像素边框，只保留纯地图
    # stitcher = SmartMinimapStitcher(
    #     x1=18, y1=68,
    #     x2=237, y2=148
    # )
    stitcher = SmartMinimapStitcher(
        x1=1081, y1=33,
        x2=1261, y2=176,
        name = "Path of Exile 2"
    )

    stitcher.run()

