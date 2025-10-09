import time
import win32gui
import win32con
import win32api

def press_key_advanced(key_code, hwnd, duration=0.05):
    """改进的按键发送"""
    try:
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.05)

        win32api.keybd_event(key_code, 0, 0, 0)
        time.sleep(duration)
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

    except Exception as e:
        print(f"  按键发送失败: {e}")


def press_crtl_key(hwnd):
    """按CTRL键 - 放技能"""
    press_key_advanced(win32con.VK_CONTROL, hwnd)
    print("  🩸 [ACTION] 按下 CTRL 键 - 放技能")


def press_del_key(hwnd):
    """按DEL键 - 补血"""
    press_key_advanced(win32con.VK_DELETE, hwnd)
    print("  🩸 [ACTION] 按下 DEL 键 - 补血")


def press_end_key(hwnd):
    press_key_advanced(win32con.VK_END, hwnd)
    print("  🩸 [ACTION] 按下 END 键 - 防御")
    time.sleep(1)
    press_key_advanced(win32con.VK_HOME, hwnd)
    print("  🩸 [ACTION] 按下 HOME 键 - 加攻")
    # time.sleep(1)
    # VK_PRIOR = 0x21
    # press_key_advanced(VK_PRIOR, hwnd)
    # print("  🩸 [ACTION] 按下 HOME 键 - 魔防")


def press_page_down_key(hwnd):
    """按PAGE DOWN键 - 补蓝"""
    VK_NEXT = 0x22
    press_key_advanced(VK_NEXT, hwnd)
    print("  💧 [ACTION] 按下 PAGE DOWN 键 - 补蓝")


def press_left_key(hwnd, duration=0.2):
    """按左方向键 - 向左移动"""
    press_key_advanced(win32con.VK_LEFT, hwnd, duration)
    print(f"  ⬅️ [ACTION] 向左移动 {duration}秒")


def press_right_key(hwnd, duration=0.2):
    """按右方向键 - 向右移动"""
    press_key_advanced(win32con.VK_RIGHT, hwnd, duration)
    print(f"  ➡️ [ACTION] 向右移动 {duration}秒")


def press_z_key(hwnd):
    """按Z键 - 拾取物品"""
    press_key_advanced(ord('Z'), hwnd)
    print("  📦 [ACTION] 按下 Z 键 - 拾取物品")

def press_up(hwnd):
    """爬绳子"""
    press_key_advanced(win32con.VK_MENU, hwnd)  # Alt down
    time.sleep(0.5)
    press_key_advanced(win32con.VK_MENU, hwnd)
    press_key_advanced(win32con.VK_UP, hwnd, 2.5)  # Up arrow
    time.sleep(0.1)
    press_key_advanced(win32con.VK_RIGHT, hwnd, 1)
    print("  📦 [ACTION] 跳跃")

def press_jump(hwnd):
    """逃脱卡死"""
    # press_key_advanced(win32con.VK_RIGHT, hwnd, 0.5)
    # time.sleep(0.1)
    # press_key_advanced(win32con.VK_MENU, hwnd)
    press_key_advanced(win32con.VK_UP, hwnd, 2.5)

    print("  📦 [ACTION] 逃脱卡死")

def auto_pickup_items(hwnd, pickup_count=5):
    """
    自动拾取物品：连续按Z键
    pickup_count: 拾取次数
    """
    print(f"  🎯 开始拾取物品（{pickup_count}次）")
    for i in range(pickup_count):
        press_z_key(hwnd)