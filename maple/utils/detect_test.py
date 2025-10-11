import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import ctypes
import time
import os
from PIL import ImageGrab


class ObjectDetectionDebugger:
    def __init__(self, window_title="MapleStory Worlds-Old School Maple"):
        self.window_title = window_title
        self.hwnd = None

        # 检测模板路径
        self.monster_templates_dir = r"C:\Repo\D4\monsters"
        self.item_templates_dir = r"C:\Repo\D4\items"
        self.character_template_path = r"C:\Repo\D4\character_name.png"

        # 模板数据
        self.monster_templates = []
        self.item_templates = []
        self.character_template = None

        # 小地图追踪
        self.minimap_region = None

        # 检测参数（可调节）
        self.monster_threshold = 0.6
        self.item_threshold = 0.7
        self.character_threshold = 0.6

        # 显示设置
        self.show_confidence = True
        self.show_names = True
        self.show_minimap = True

        # 颜色设置
        self.colors = {
            'character': (0, 255, 255),  # 黄色
            'monster': (0, 0, 255),  # 红色
            'item': (0, 255, 0),  # 绿色
            'minimap_player': (255, 0, 255)  # 紫色
        }

        self.initialize()

    def initialize(self):
        """初始化系统"""
        print("🔍 初始化对象检测调试器...")

        try:
            # 查找游戏窗口
            self.hwnd = win32gui.FindWindow(None, self.window_title)
            if not self.hwnd:
                raise Exception(f"未找到窗口: {self.window_title}")
            print("✅ 找到游戏窗口")

            # 校准小地图
            self.calibrate_minimap()

            # 加载模板
            self.load_all_templates()

            print("✅ 初始化完成")

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def calibrate_minimap(self):
        """校准小地图区域"""
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)

        minimap_left = left + 15
        minimap_top = top + 65
        minimap_right = left + 280
        minimap_bottom = top + 200

        self.minimap_region = (minimap_left, minimap_top, minimap_right, minimap_bottom)
        print(f"🗺️ 小地图区域: {self.minimap_region}")

    def load_template_safe(self, template_path, template_name):
        """安全加载模板"""
        try:
            if not os.path.exists(template_path):
                return None

            template = cv2.imread(template_path)
            if template is None:
                print(f"⚠️ 无法读取: {template_path}")
                return None

            if template.shape[0] == 0 or template.shape[1] == 0:
                print(f"⚠️ 模板尺寸无效: {template_path}")
                return None

            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            print(f"✅ 加载模板: {template_name}")
            return {
                'name': template_name,
                'template': template,
                'gray': template_gray,
                'path': template_path
            }

        except Exception as e:
            print(f"❌ 加载模板失败 {template_path}: {e}")
            return None

    def load_all_templates(self):
        """加载所有模板"""
        # 创建目录
        os.makedirs(self.monster_templates_dir, exist_ok=True)
        os.makedirs(self.item_templates_dir, exist_ok=True)

        # 加载怪物模板
        monster_count = 0
        if os.path.exists(self.monster_templates_dir):
            for filename in os.listdir(self.monster_templates_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    template_path = os.path.join(self.monster_templates_dir, filename)
                    template_data = self.load_template_safe(template_path, filename)
                    if template_data:
                        self.monster_templates.append(template_data)
                        monster_count += 1

        # 加载物品模板
        item_count = 0
        if os.path.exists(self.item_templates_dir):
            for filename in os.listdir(self.item_templates_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    template_path = os.path.join(self.item_templates_dir, filename)
                    template_data = self.load_template_safe(template_path, filename)
                    if template_data:
                        self.item_templates.append(template_data)
                        item_count += 1

        # 加载人物模板
        character_loaded = False
        if os.path.exists(self.character_template_path):
            character_template = self.load_template_safe(self.character_template_path, "character_name")
            if character_template:
                self.character_template = character_template
                character_loaded = True

        print(f"\n📊 模板加载摘要:")
        print(f"  🦄 怪物模板: {monster_count} 个")
        print(f"  💎 物品模板: {item_count} 个")
        print(f"  👤 人物模板: {'1' if character_loaded else '0'} 个")

    def capture_window(self):
        """截取游戏窗口"""
        try:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width, height = right - left, bottom - top

            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
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

        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def capture_minimap(self):
        """截取小地图"""
        try:
            minimap_img = ImageGrab.grab(bbox=self.minimap_region)
            minimap_cv = cv2.cvtColor(np.array(minimap_img), cv2.COLOR_RGB2BGR)
            return minimap_cv
        except Exception as e:
            print(f"小地图截取失败: {e}")
            return None

    def find_yellow_dot(self, minimap_img):
        """在小地图中找到黄色点"""
        if minimap_img is None:
            return None

        try:
            hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        except Exception as e:
            print(f"黄点检测失败: {e}")
        return None

    def detect_character(self, img):
        """检测人物"""
        if self.character_template is None or img is None:
            return []

        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            template_gray = self.character_template['gray']

            matches = []

            # 多尺度匹配
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                h, w = template_gray.shape
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)

                if scaled_w <= 0 or scaled_h <= 0:
                    continue
                if scaled_h > img_gray.shape[0] or scaled_w > img_gray.shape[1]:
                    continue

                scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
                result = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)

                locations = np.where(result >= self.character_threshold)

                for pt in zip(*locations[::-1]):
                    center_x = pt[0] + scaled_w // 2
                    center_y = pt[1] + scaled_h // 2
                    character_y = center_y + scaled_h + 20  # 估算人物位置

                    matches.append({
                        'type': 'character',
                        'name': 'Player',
                        'position': pt,
                        'center': (center_x, center_y),
                        'character_pos': (center_x, character_y),
                        'confidence': result[pt[1], pt[0]],
                        'size': (scaled_w, scaled_h),
                        'scale': scale
                    })

            # 去重（相近的检测结果）
            filtered_matches = []
            for match in matches:
                is_duplicate = False
                for existing in filtered_matches:
                    distance = np.sqrt((match['center'][0] - existing['center'][0]) ** 2 +
                                       (match['center'][1] - existing['center'][1]) ** 2)
                    if distance < 30:  # 30像素内认为是重复
                        if match['confidence'] > existing['confidence']:
                            filtered_matches.remove(existing)
                        else:
                            is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_matches.append(match)

            return filtered_matches

        except Exception as e:
            print(f"人物检测失败: {e}")
            return []

    def detect_objects(self, img, templates, obj_type, threshold):
        """检测物体（怪物或物品）"""
        if img is None or len(templates) == 0:
            return []

        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_objects = []

            for template_info in templates:
                if template_info is None or template_info.get('gray') is None:
                    continue

                template_gray = template_info['gray']

                for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
                    h, w = template_gray.shape
                    resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))

                    if (resized_template.shape[0] > img_gray.shape[0] or
                            resized_template.shape[1] > img_gray.shape[1]):
                        continue

                    result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= threshold)

                    for pt in zip(*locations[::-1]):
                        detected_objects.append({
                            'type': obj_type,
                            'name': template_info['name'],
                            'position': pt,
                            'center': (pt[0] + resized_template.shape[1] // 2,
                                       pt[1] + resized_template.shape[0] // 2),
                            'confidence': result[pt[1], pt[0]],
                            'size': (resized_template.shape[1], resized_template.shape[0]),
                            'scale': scale
                        })

            # 去重
            filtered_objects = []
            for obj in detected_objects:
                is_duplicate = False
                for existing in filtered_objects:
                    if obj['name'] == existing['name']:
                        distance = np.sqrt((obj['center'][0] - existing['center'][0]) ** 2 +
                                           (obj['center'][1] - existing['center'][1]) ** 2)
                        if distance < 40:  # 40像素内认为是重复
                            if obj['confidence'] > existing['confidence']:
                                filtered_objects.remove(existing)
                            else:
                                is_duplicate = True
                            break
                if not is_duplicate:
                    filtered_objects.append(obj)

            return filtered_objects

        except Exception as e:
            print(f"{obj_type}检测失败: {e}")
            return []

    def draw_detection_results(self, img, detections, minimap_pos=None):
        """在图像上绘制检测结果"""
        if img is None:
            return img

        vis_img = img.copy()

        # 绘制检测到的对象
        for detection in detections:
            obj_type = detection['type']
            color = self.colors.get(obj_type, (255, 255, 255))

            # 绘制边框
            pos = detection['position']
            size = detection['size']
            cv2.rectangle(vis_img, pos, (pos[0] + size[0], pos[1] + size[1]), color, 2)

            # 绘制中心点
            center = detection['center']
            cv2.circle(vis_img, center, 5, color, -1)

            # 绘制标签
            if self.show_names:
                label = detection['name']
                if self.show_confidence:
                    label += f" ({detection['confidence']:.2f})"

                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # 绘制文本背景
                cv2.rectangle(vis_img,
                              (pos[0], pos[1] - text_height - baseline - 5),
                              (pos[0] + text_width, pos[1] - baseline),
                              color, -1)

                # 绘制文本
                cv2.putText(vis_img, label, (pos[0], pos[1] - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 绘制统计信息
        self.draw_statistics(vis_img, detections, minimap_pos)

        return vis_img

    def draw_statistics(self, img, detections, minimap_pos):
        """绘制统计信息"""
        h, w = img.shape[:2]

        # 统计各类型数量
        stats = {'character': 0, 'monster': 0, 'item': 0}
        for detection in detections:
            obj_type = detection['type']
            if obj_type in stats:
                stats[obj_type] += 1

        # 绘制信息面板
        panel_width = 300
        panel_height = 150
        panel_x = w - panel_width - 10
        panel_y = 10

        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 绘制边框
        cv2.rectangle(img, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (255, 255, 255), 2)

        # 绘制标题
        cv2.putText(img, "Detection Statistics",
                    (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制统计信息
        y_offset = 50
        line_height = 20

        # 人物信息
        color = self.colors['character']
        cv2.putText(img, f"Characters: {stats['character']}",
                    (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height

        # 怪物信息
        color = self.colors['monster']
        cv2.putText(img, f"Monsters: {stats['monster']}",
                    (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height

        # 物品信息
        color = self.colors['item']
        cv2.putText(img, f"Items: {stats['item']}",
                    (panel_x + 10, panel_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height

        # 小地图位置
        if minimap_pos:
            cv2.putText(img, f"Position: {minimap_pos}",
                        (panel_x + 10, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['minimap_player'], 1)
        else:
            cv2.putText(img, "Position: Unknown",
                        (panel_x + 10, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    def create_minimap_visualization(self, minimap_img, player_pos):
        """创建小地图可视化"""
        if minimap_img is None:
            return np.zeros((200, 265, 3), dtype=np.uint8)

        # 放大小地图
        scale_factor = 2
        minimap_large = cv2.resize(minimap_img, None, fx=scale_factor, fy=scale_factor,
                                   interpolation=cv2.INTER_NEAREST)

        # 标记玩家位置
        if player_pos:
            scaled_pos = (player_pos[0] * scale_factor, player_pos[1] * scale_factor)
            cv2.circle(minimap_large, scaled_pos, 8, self.colors['minimap_player'], 2)
            cv2.putText(minimap_large, "Player",
                        (scaled_pos[0] + 10, scaled_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['minimap_player'], 1)

        # 绘制中心点参考
        center_x = minimap_large.shape[1] // 2
        center_y = minimap_large.shape[0] // 2
        cv2.circle(minimap_large, (center_x, center_y), 3, (255, 255, 255), 1)
        cv2.putText(minimap_large, "Center", (center_x + 5, center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return minimap_large

    def handle_keyboard_input(self, key):
        """处理键盘输入调整参数"""
        if key == ord('1'):
            self.monster_threshold = max(0.1, self.monster_threshold - 0.05)
            print(f"🦄 怪物阈值: {self.monster_threshold:.2f}")
        elif key == ord('2'):
            self.monster_threshold = min(0.95, self.monster_threshold + 0.05)
            print(f"🦄 怪物阈值: {self.monster_threshold:.2f}")
        elif key == ord('3'):
            self.item_threshold = max(0.1, self.item_threshold - 0.05)
            print(f"💎 物品阈值: {self.item_threshold:.2f}")
        elif key == ord('4'):
            self.item_threshold = min(0.95, self.item_threshold + 0.05)
            print(f"💎 物品阈值: {self.item_threshold:.2f}")
        elif key == ord('5'):
            self.character_threshold = max(0.1, self.character_threshold - 0.05)
            print(f"👤 人物阈值: {self.character_threshold:.2f}")
        elif key == ord('6'):
            self.character_threshold = min(0.95, self.character_threshold + 0.05)
            print(f"👤 人物阈值: {self.character_threshold:.2f}")
        elif key == ord('c') or key == ord('C'):
            self.show_confidence = not self.show_confidence
            print(f"📊 显示置信度: {'开启' if self.show_confidence else '关闭'}")
        elif key == ord('n') or key == ord('N'):
            self.show_names = not self.show_names
            print(f"🏷️ 显示名称: {'开启' if self.show_names else '关闭'}")
        elif key == ord('m') or key == ord('M'):
            self.show_minimap = not self.show_minimap
            print(f"🗺️ 显示小地图: {'开启' if self.show_minimap else '关闭'}")

    def run_debug_mode(self):
        """运行调试模式"""
        print("🔍 启动对象检测调试模式")
        print("=" * 60)
        print("📋 控制说明:")
        print("  1/2 - 调整怪物检测阈值 (↓/↑)")
        print("  3/4 - 调整物品检测阈值 (↓/↑)")
        print("  5/6 - 调整人物检测阈值 (↓/↑)")
        print("  C   - 切换置信度显示")
        print("  N   - 切换名称显示")
        print("  M   - 切换小地图显示")
        print("  ESC - 退出程序")
        print("=" * 60)

        frame_count = 0
        fps_timer = time.time()

        try:
            while True:
                frame_start = time.time()
                frame_count += 1

                # 截取游戏画面
                screenshot = self.capture_window()
                if screenshot is None:
                    print("❌ 截图失败")
                    time.sleep(0.5)
                    continue

                # 获取小地图玩家位置
                minimap_img = self.capture_minimap()
                player_minimap_pos = self.find_yellow_dot(minimap_img)

                # 检测所有对象
                all_detections = []

                # 检测人物
                characters = self.detect_character(screenshot)
                all_detections.extend(characters)

                # 检测怪物
                monsters = self.detect_objects(screenshot, self.monster_templates,
                                               'monster', self.monster_threshold)
                all_detections.extend(monsters)

                # 检测物品
                items = self.detect_objects(screenshot, self.item_templates,
                                            'item', self.item_threshold)
                all_detections.extend(items)

                # 绘制检测结果
                vis_screenshot = self.draw_detection_results(screenshot, all_detections, player_minimap_pos)

                # 调整显示尺寸
                display_height = 800
                scale = display_height / vis_screenshot.shape[0]
                display_width = int(vis_screenshot.shape[1] * scale)
                vis_resized = cv2.resize(vis_screenshot, (display_width, display_height))

                # 显示主窗口
                cv2.imshow("Object Detection Debug - Main View", vis_resized)

                # 显示小地图（如果启用）
                if self.show_minimap:
                    minimap_vis = self.create_minimap_visualization(minimap_img, player_minimap_pos)
                    cv2.imshow("Object Detection Debug - Minimap", minimap_vis)

                # 计算FPS
                if frame_count % 30 == 0:  # 每30帧计算一次FPS
                    current_time = time.time()
                    fps = 30 / (current_time - fps_timer)
                    fps_timer = current_time

                    # 输出统计信息
                    char_count = len([d for d in all_detections if d['type'] == 'character'])
                    monster_count = len([d for d in all_detections if d['type'] == 'monster'])
                    item_count = len([d for d in all_detections if d['type'] == 'item'])

                    print(f"[Frame {frame_count:04d}] FPS: {fps:.1f} | "
                          f"👤:{char_count} 🦄:{monster_count} 💎:{item_count} | "
                          f"位置:{player_minimap_pos}")

                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    print("👋 退出调试模式")
                    break
                elif key != 255:
                    self.handle_keyboard_input(key)

                # 控制帧率
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / 30  # 30 FPS
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        except Exception as e:
            print(f"❌ 调试模式错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()


def create_sample_templates():
    """创建示例模板文件夹"""
    dirs = [
        r"C:\Repo\D4\monsters",
        r"C:\Repo\D4\items"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("📁 已创建模板文件夹:")
    print("  C:\\Repo\\D4\\monsters\\ - 放入怪物图片")
    print("  C:\\Repo\\D4\\items\\ - 放入物品图片")
    print("  C:\\Repo\\D4\\character_name.png - 人物名字模板 (可选)")


if __name__ == "__main__":
    print("=" * 80)
    print("🔍 MapleStory Worlds 对象检测调试器 v1.0")
    print("   by RYAN4238 - 2025-01-10")
    print("=" * 80)
    print("\n🎯 功能特性:")
    print("  ✅ 实时检测并可视化人物、怪物、物品")
    print("  ✅ 小地图玩家位置追踪")
    print("  ✅ 可调节检测阈值和显示选项")
    print("  ✅ 实时FPS和统计信息")
    print("  ✅ 纯调试模式，无任何自动化操作")
    print("=" * 80)

    # 创建模板文件夹
    create_sample_templates()

    try:
        debugger = ObjectDetectionDebugger()

        print("\n🚀 启动调试器...")
        print("准备就绪，按任意键开始或Ctrl+C退出")
        input()

        debugger.run_debug_mode()

    except KeyboardInterrupt:
        print("\n👋 用户取消启动")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback

        traceback.print_exc()
        input("按回车键退出...")