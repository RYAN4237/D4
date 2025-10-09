import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import ctypes
import time
import pytesseract
import re
import random

from utils.map import *
from utils.hp import extract_status_bars_precise, calculate_bar_fill_ratio
from utils.key import *
from utils.move import *

# 如果需要指定tesseract路径，取消下面这行的注释并修改路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

PW_RENDERFULLCONTENT = 0x00000002
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\tesseract-ocr\tesseract.exe'


def get_game_window(title="MapleStory Worlds-Old School Maple"):
    """获取游戏窗口句柄"""
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise Exception(f"未找到窗口: {title}")
    return hwnd


def capture_window(hwnd):
    """截取窗口图像"""
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), PW_RENDERFULLCONTENT)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img_bgr


def auto_pickup_items(hwnd, pickup_count=5):
    """
    自动拾取物品：连续按Z键
    pickup_count: 拾取次数
    """
    print(f"  🎯 开始拾取物品（{pickup_count}次）")
    for i in range(pickup_count):
        press_z_key(hwnd)
        time.sleep(0.2)


def auto_hp_mp_monitor(hp_threshold=70, mp_threshold=60, enable_auto_press=True,
                       move_interval=15, pickup_interval=10, buff_interval=60, left=0, right=800, key_pos=[]):
    """
    自动HP/MP监控 + 随机移动 + 拾取
    """
    try:
        hwnd = get_game_window()
        print("找到游戏窗口: MapleStory Worlds-Old School Maple")
        print(f"\n配置:")
        print(f"  HP阈值: {hp_threshold}%")
        print(f"  MP阈值: {mp_threshold}%")
        print(f"  自动按键: {'启用' if enable_auto_press else '禁用(调试模式)'}")
        print(f"  随机移动间隔: {move_interval}秒")
        print(f"  拾取物品间隔: {pickup_interval}秒")
        print(f"  施加buff间隔: {buff_interval}秒")
        print("\n开始监控...\n")

        last_hp_press = 0
        last_mp_press = 0
        last_move = 0
        last_pickup = 0
        last_buff = 0
        press_cooldown = 2.0

        frame_count = 0
        last_detected_name = None
        tracker = MinimapPositionTracker()
        tracker.calibrate_minimap_region()

        while True:
            frame_count += 1
            img = capture_window(hwnd)

            # 提取HP和MP区域
            hp_region, mp_region, status_region, region_pos = extract_status_bars_precise(img)

            # 计算百分比
            hp_pct, hp_mask, hp_start, hp_end = calculate_bar_fill_ratio(hp_region, 'red')
            mp_pct, mp_mask, mp_start, mp_end = calculate_bar_fill_ratio(mp_region, 'blue')

            current_time = time.time()

            # 每2帧打印一次状态
            if frame_count % 6 == 0:
                status = f"[Frame {frame_count:04d}] HP: {hp_pct:5.1f}%  MP: {mp_pct:5.1f}%"

                if last_detected_name:
                    status += f" | 👤 {last_detected_name}"

                if hp_pct < hp_threshold:
                    status += " 🩸LOW HP"
                if mp_pct < mp_threshold:
                    status += " 💧LOW MP"

                print(status)

                if enable_auto_press:
                    press_crtl_key(hwnd)

            # 检查HP并执行动作
            if hp_pct < hp_threshold and (current_time - last_hp_press) > press_cooldown:
                if enable_auto_press:
                    press_del_key(hwnd)
                    last_hp_press = current_time

            # 检查MP并执行动作
            if mp_pct < mp_threshold and (current_time - last_mp_press) > press_cooldown:
                if enable_auto_press:
                    press_page_down_key(hwnd)
                    last_mp_press = current_time


            # 定期执行随机移动
            if enable_auto_press and (current_time - last_move) > move_interval:
                print(f"\n⏰ 执行定期随机移动...")

                traversal_map(left, right, tracker, hwnd, key_pos=key_pos, slice=10)
                print()

            # # 定期拾取物品
            if enable_auto_press and (current_time - last_pickup) > pickup_interval:
                print(f"\n⏰ 执行定期拾取...")
                auto_pickup_items(hwnd, pickup_count=5)
                last_pickup = current_time
                print()
            #
            # # buff
            if enable_auto_press and (current_time - last_buff) > buff_interval:
                print("施加buff")
                press_end_key(hwnd)
                last_buff = current_time
                print()



    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("MapleStory 随机移动监控脚本 v4.1")
    print("=" * 60)
    print("\n功能:")
    print("✓ 使用OCR识别人物名牌")
    print("✓ 精确检测HP/MP百分比并自动补给")
    print("✓ 自动放技能（CTRL）")
    print("✓ 每15秒随机左右移动10次")
    print("✓ 定期自动拾取物品（Z键）")
    print("✓ 可视化人物识别框和名字")
    print("=" * 60)

    # 配置参数
    HP_THRESHOLD = 50
    MP_THRESHOLD = 50
    DEBUG_MODE = False
    MOVE_INTERVAL = 15  # 随机移动间隔
    PICKUP_INTERVAL = 20

    print(f"\n当前配置:")
    print(f"  HP阈值: {HP_THRESHOLD}%")
    print(f"  MP阈值: {MP_THRESHOLD}%")
    print(f"  随机移动间隔: {MOVE_INTERVAL}秒")
    print(f"  拾取间隔: {PICKUP_INTERVAL}秒")
    print(f"  调试模式: {'开启' if DEBUG_MODE else '关闭'}")
    print(f"\n请确保已安装 Tesseract OCR!")

    auto_hp_mp_monitor(
        hp_threshold=HP_THRESHOLD,
        mp_threshold=MP_THRESHOLD,
        enable_auto_press=not DEBUG_MODE
    )