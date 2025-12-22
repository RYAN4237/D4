import cv2

from poe2.capture import *
from poe2.actions import *

def attack_nearby_enemies(driver: Driver, capture: CaptureScreen):
    """
    检测并攻击附近的敌人。

    参数:
      - driver: Driver 对象，用于发送按键指令
      - capture: Capture 对象，用于截图和图像处理
      - attack_radius: 攻击半径，单位为像素
    """
    img = capture.capture(220,47,1009,651, 9, 32)  # 截取游戏区域
    # 转换为HSV色彩空间(更适合颜色检测)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    # 红色在HSV中有两个范围(因为红色在色相环的两端)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合并两个mask
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作: 去除噪点
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    # 寻找轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选血条(通常是细长的矩形)
    health_bars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 根据血条特征筛选: 宽度大于高度,面积适中
        aspect_ratio = w / float(h) if h > 0 else 0
        area = cv2.contourArea(contour)

        # 血条通常是横向的长条形(宽高比 > 2)且面积适中
        if aspect_ratio > 2 and 50 < area < 5000:
            # 在原图上绘制检测结果
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            driver.move_and_click(x + 220, y + 47, capture.win_x + 9, capture.win_y + 32)  # 考虑截图偏移和窗口偏移
            # time.sleep(0.1)  # 等待一段时间以避免过快点击

    # cv2.namedWindow("Enemy Health Bars", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("Enemy Health Bars", 1000, 500)
    # cv2.imshow("Enemy Health Bars", img)