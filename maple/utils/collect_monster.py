import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import ctypes
import os
import time
from PIL import ImageGrab


class TemplateCreator:
    def __init__(self, window_title="MapleStory Worlds-Old School Maple"):
        self.window_title = window_title
        self.hwnd = None

        # 鼠标选择相关变量
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.current_img = None
        self.display_img = None

        # 模板保存路径
        self.monsters_dir = r"C:\Repo\D4\monsters"
        self.items_dir = r"C:\Repo\D4\items"

        # 创建目录
        os.makedirs(self.monsters_dir, exist_ok=True)
        os.makedirs(self.items_dir, exist_ok=True)

        # 计数器
        self.monster_count = len([f for f in os.listdir(self.monsters_dir) if f.endswith('.png')])
        self.item_count = len([f for f in os.listdir(self.items_dir) if f.endswith('.png')])

    def capture_window(self):
        """截取游戏窗口"""
        if not self.hwnd:
            self.hwnd = win32gui.FindWindow(None, self.window_title)
            if not self.hwnd:
                raise Exception(f"未找到窗口: {self.window_title}")

        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width, height = right - left, bottom - top

        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # 使用 PrintWindow 获取窗口内容
        ctypes.windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
            (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 清理资源
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        return img_bgr

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                # 实时显示选择框
                self.display_img = self.current_img.copy()
                cv2.rectangle(self.display_img, self.start_point, self.end_point, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            # 绘制最终选择框
            cv2.rectangle(self.display_img, self.start_point, self.end_point, (0, 255, 0), 2)

    def get_selection_area(self):
        """获取选择的区域"""
        if (self.start_point == (-1, -1) or self.end_point == (-1, -1) or
                self.start_point == self.end_point):
            return None

        # 确保坐标正确（处理反向选择）
        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])

        # 检查区域大小
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        return (x1, y1, x2, y2)

    def save_template(self, template_img, template_type, custom_name=None):
        """保存模板图片"""
        if template_type == 'monster':
            if custom_name:
                filename = f"{custom_name}.png"
            else:
                self.monster_count += 1
                filename = f"monster_{self.monster_count:03d}.png"
            filepath = os.path.join(self.monsters_dir, filename)
        else:  # item
            if custom_name:
                filename = f"{custom_name}.png"
            else:
                self.item_count += 1
                filename = f"item_{self.item_count:03d}.png"
            filepath = os.path.join(self.items_dir, filename)

        cv2.imwrite(filepath, template_img)
        print(f"✅ 已保存模板: {filepath}")
        return filename

    def create_templates_interactive(self):
        """交互式创建模板"""
        print("🎮 模板制作工具启动")
        print("=" * 50)
        print("📋 操作说明:")
        print("  1. 按 'S' - 截取新的游戏画面")
        print("  2. 鼠标拖拽 - 选择模板区域")
        print("  3. 按 'M' - 保存为怪物模板")
        print("  4. 按 'I' - 保存为物品模板")
        print("  5. 按 'C' - 清除当前选择")
        print("  6. 按 'Q' - 退出程序")
        print("=" * 50)

        # 初始截图
        try:
            self.current_img = self.capture_window()
            self.display_img = self.current_img.copy()
            print(f"📸 已截取游戏画面 ({self.current_img.shape[1]}x{self.current_img.shape[0]})")
        except Exception as e:
            print(f"❌ 截图失败: {e}")
            return

        # 创建窗口并设置鼠标回调
        cv2.namedWindow("Template Creator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Template Creator", self.mouse_callback)

        # 调整窗口大小以适应屏幕
        screen_height = 1080  # 假设屏幕高度
        if self.current_img.shape[0] > screen_height - 100:
            scale = (screen_height - 100) / self.current_img.shape[0]
            new_width = int(self.current_img.shape[1] * scale)
            new_height = int(self.current_img.shape[0] * scale)
            cv2.resizeWindow("Template Creator", new_width, new_height)

        while True:
            # 显示当前图像
            cv2.imshow("Template Creator", self.display_img)

            # 获取按键
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("👋 退出模板制作工具")
                break

            elif key == ord('s') or key == ord('S'):
                # 重新截图
                try:
                    print("📸 正在截取新画面...")
                    self.current_img = self.capture_window()
                    self.display_img = self.current_img.copy()
                    self.start_point = (-1, -1)
                    self.end_point = (-1, -1)
                    print("✅ 截图完成")
                except Exception as e:
                    print(f"❌ 截图失败: {e}")

            elif key == ord('m') or key == ord('M'):
                # 保存怪物模板
                area = self.get_selection_area()
                if area:
                    x1, y1, x2, y2 = area
                    template = self.current_img[y1:y2, x1:x2]

                    # 询问自定义名称
                    print("\n🦁 保存怪物模板")
                    name_input = input("输入模板名称 (直接回车使用默认名称): ").strip()
                    custom_name = name_input if name_input else None

                    filename = self.save_template(template, 'monster', custom_name)

                    # 显示预览
                    preview = cv2.resize(template, (150, 150)) if min(template.shape[:2]) > 150 else template
                    cv2.imshow(f"Monster Template: {filename}", preview)

                    # 清除选择
                    self.display_img = self.current_img.copy()
                    self.start_point = (-1, -1)
                    self.end_point = (-1, -1)
                else:
                    print("❌ 请先选择一个区域")

            elif key == ord('i') or key == ord('I'):
                # 保存物品模板
                area = self.get_selection_area()
                if area:
                    x1, y1, x2, y2 = area
                    template = self.current_img[y1:y2, x1:x2]

                    # 询问自定义名称
                    print("\n💎 保存物品模板")
                    name_input = input("输入模板名称 (直接回车使用默认名称): ").strip()
                    custom_name = name_input if name_input else None

                    filename = self.save_template(template, 'item', custom_name)

                    # 显示预览
                    preview = cv2.resize(template, (150, 150)) if min(template.shape[:2]) > 150 else template
                    cv2.imshow(f"Item Template: {filename}", preview)

                    # 清除选择
                    self.display_img = self.current_img.copy()
                    self.start_point = (-1, -1)
                    self.end_point = (-1, -1)
                else:
                    print("❌ 请先选择一个区域")

            elif key == ord('c') or key == ord('C'):
                # 清除选择
                self.display_img = self.current_img.copy()
                self.start_point = (-1, -1)
                self.end_point = (-1, -1)
                print("🧹 已清除选择")

            # 实时更新选择框显示
            if self.drawing:
                temp_img = self.current_img.copy()
                cv2.rectangle(temp_img, self.start_point, self.end_point, (0, 255, 0), 2)
                self.display_img = temp_img

        cv2.destroyAllWindows()

        # 显示统计
        print("\n📊 模板制作完成!")
        print(f"🦁 怪物模板: {len([f for f in os.listdir(self.monsters_dir) if f.endswith('.png')])} 个")
        print(f"💎 物品模板: {len([f for f in os.listdir(self.items_dir) if f.endswith('.png')])} 个")


def batch_template_creator():
    """批量模板制作模式"""
    creator = TemplateCreator()

    print("🔄 批量模板制作模式")
    print("这个模式会连续截图，让你快速制作多个模板")

    template_type = input("选择模板类型 (M=怪物, I=物品): ").upper()
    if template_type not in ['M', 'I']:
        print("❌ 无效选择")
        return

    count = 0
    while True:
        input(f"\n按回车键截取第 {count + 1} 个模板...")

        try:
            img = creator.capture_window()
            print(f"📸 已截取画面 {count + 1}")

            # 显示图像让用户选择
            cv2.namedWindow("Batch Template Creator", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Batch Template Creator", creator.mouse_callback)

            creator.current_img = img
            creator.display_img = img.copy()
            creator.start_point = (-1, -1)
            creator.end_point = (-1, -1)

            print("🖱️ 请拖拽鼠标选择区域，然后按空格键确认，按ESC跳过")

            while True:
                cv2.imshow("Batch Template Creator", creator.display_img)
                key = cv2.waitKey(1) & 0xFF

                if key == 32:  # 空格键确认
                    area = creator.get_selection_area()
                    if area:
                        x1, y1, x2, y2 = area
                        template = img[y1:y2, x1:x2]

                        name_input = input("输入模板名称 (回车使用默认): ").strip()
                        custom_name = name_input if name_input else None

                        if template_type == 'M':
                            creator.save_template(template, 'monster', custom_name)
                        else:
                            creator.save_template(template, 'item', custom_name)

                        count += 1
                    break

                elif key == 27:  # ESC跳过
                    print("⏭️ 跳过当前模板")
                    break

            # 询问是否继续
            continue_choice = input("继续制作下一个模板? (y/n): ").lower()
            if continue_choice != 'y':
                break

        except Exception as e:
            print(f"❌ 错误: {e}")
            break

    cv2.destroyAllWindows()
    print(f"✅ 批量制作完成，共创建 {count} 个模板")


if __name__ == "__main__":
    print("🛠️ MapleStory 模板制作工具")
    print("=" * 40)
    print("选择模式:")
    print("1. 交互式模板制作")
    print("2. 批量模板制作")

    choice = input("请选择模式 (1/2): ").strip()

    if choice == "1":
        creator = TemplateCreator()
        creator.create_templates_interactive()
    elif choice == "2":
        batch_template_creator()
    else:
        print("❌ 无效选择")