"""
智能小地图拼接 - 带实时监控版本
只在检测到真实移动时才拼接，避免累积误差
"""
import time
from capture import CaptureScreen
import numpy as np
import cv2
import multiprocessing as mp

# 全局共享变量（用于多进程通信）
shared_result = None
result_queue = None  # 用于进程间传递图像

class SmartMinimapStitcher:
    """
    智能小地图拼接器

    特性：
    - 累积平均消除位移误差裂纹
    - 动态边界扩展（迷雾探索）
    - 亚像素精度位移检测
    """

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        name: str = "Path of Exile 2",
        canvas_size: int = 3000,
        movement_threshold: float = 3.0,
        confidence_threshold: float = 0.25,
        diff_threshold: int = 25,
        clarity_threshold: int = 10,
        significant_weight: float = 5.0
    ):
        """
        初始化拼接器

        Args:
            x1, y1, x2, y2: 小地图截图区域
            name: 游戏窗口标题
            canvas_size: 画布尺寸（正方形）
            movement_threshold: 帧差阈值（低于此值认为没有移动）
            confidence_threshold: 相位相关置信度阈值
            diff_threshold: 像素差异阈值（判断显著变化）
            clarity_threshold: 清晰度差异阈值（判断边界扩展）
            significant_weight: 显著变化区域的累积权重倍数
        """
        # 初始化截图器
        self.jy = CaptureScreen()
        self.jy.get_hwnd(name)

        # 截图区域
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

        # 画布配置
        self.canvas_size = canvas_size
        self.canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 200
        self.canvas_accum = np.zeros((canvas_size, canvas_size, 3), dtype=np.float64)
        self.canvas_count = np.zeros((canvas_size, canvas_size), dtype=np.float64)
        self.explored_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # 当前位置（浮点精度）
        self.canvas_x = float(canvas_size // 2)
        self.canvas_y = float(canvas_size // 2)

        # 算法参数
        self.movement_threshold = movement_threshold
        self.confidence_threshold = confidence_threshold
        self.diff_threshold = diff_threshold
        self.clarity_threshold = clarity_threshold
        self.significant_weight = significant_weight

        # 统计信息
        self.frame_count = 0
        self.stitch_count = 0
        self.last_minimap = None

        print("=" * 60)
        print("智能小地图拼接工具 v3.0")
        print("=" * 60)
        print(f"截图区域: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"画布大小: {canvas_size} x {canvas_size}")
        print(f"算法参数: 移动阈值={movement_threshold}, "
              f"置信度={confidence_threshold}, 差异阈值={diff_threshold}")
        print("=" * 60)

    def capture_minimap(self):
        """截取小地图"""
        return self.jy.capture(self.x1, self.y1, self.x2, self.y2, 1, 1)

    def detect_movement(self, img1, img2):
        """
        检测两帧之间的位移（亚像素精度）

        使用 HSV 过滤 + 自适应阈值 + 相位相关

        Returns:
            tuple: (has_moved, dx, dy, confidence)
                - has_moved: bool, 是否检测到有效移动
                - dx, dy: float, 位移量（保留亚像素精度）
                - confidence: float, 检测置信度
        """
        # HSV 颜色空间过滤（提取地图特征）
        hsv_lower = np.array([20, 60, 150], dtype=np.uint8)
        hsv_upper = np.array([130, 190, 220], dtype=np.uint8)

        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img1_mask = cv2.inRange(img1_hsv, hsv_lower, hsv_upper)
        img1_filtered = cv2.bitwise_and(img1, img1, mask=img1_mask)

        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        img2_mask = cv2.inRange(img2_hsv, hsv_lower, hsv_upper)
        img2_filtered = cv2.bitwise_and(img2, img2, mask=img2_mask)

        # 转灰度 + 自适应阈值 + 中值滤波
        gray1 = cv2.cvtColor(img1_filtered, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.adaptiveThreshold(
            gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1
        )
        gray1 = cv2.medianBlur(gray1, 3)

        gray2 = cv2.cvtColor(img2_filtered, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.adaptiveThreshold(
            gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1
        )
        gray2 = cv2.medianBlur(gray2, 3)

        # 计算帧差
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = np.mean(diff)

        if mean_diff < self.movement_threshold:
            return False, 0.0, 0.0, 0.0

        # 相位相关计算位移
        try:
            shift, response = cv2.phaseCorrelate(
                np.float32(gray1), np.float32(gray2)
            )

            dx, dy = shift[0], shift[1]

            # 位移太小
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                return False, 0.0, 0.0, response

            # 置信度太低
            if response < self.confidence_threshold:
                return False, 0.0, 0.0, response

            # 位移太大（可能误判）
            if abs(dx) > 50 or abs(dy) > 50:
                return False, 0.0, 0.0, response

            return True, dx, dy, response

        except Exception as e:
            print(f"  警告: 位移检测失败 - {e}")
            return False, 0.0, 0.0, 0.0

    def stitch(self, minimap):
        """将小地图拼接到画布上（完全覆盖模式，避免色差）"""
        h, w = minimap.shape[:2]

        # 计算在画布上的位置（四舍五入并转换为整数）
        y1 = int(round(self.canvas_y))
        x1 = int(round(self.canvas_x))
        y2 = y1 + h
        x2 = x1 + w

        # 裁剪到画布范围并计算源图对应的切片
        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(self.canvas_size, y2)
        x2c = min(self.canvas_size, x2)

        # 目标区域尺寸
        th = y2c - y1c
        tw = x2c - x1c
        if th <= 0 or tw <= 0:
            # 完全超出画布
            print(f"  ⚠️ 警告: 小地图完全超出画布范围 ({x1},{y1})-({x2},{y2})")
            return False

        # 源图对应切片（处理部分越界情况）
        sy1 = max(0, - (y1 - y1c))
        sx1 = max(0, - (x1 - x1c))
        sy2 = sy1 + th
        sx2 = sx1 + tw

        # 赋值（处理尺寸不一致）
        self.canvas[y1c:y2c, x1c:x2c] = minimap[sy1:sy2, sx1:sx2]
        return True

    def stitch_incremental(self, minimap):
        h, w = minimap.shape[:2]

        # 使用整数并裁剪到画布边界（与 stitch 保持一致的安全处理）
        y1 = int(round(self.canvas_y))
        x1 = int(round(self.canvas_x))
        y2 = y1 + h
        x2 = x1 + w

        # 裁剪目标和源切片
        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(self.canvas_size, y2)
        x2c = min(self.canvas_size, x2)

        th = y2c - y1c
        tw = x2c - x1c
        if th <= 0 or tw <= 0:
            return False

        sy1 = max(0, - (y1 - y1c))
        sx1 = max(0, - (x1 - x1c))
        sy2 = sy1 + th
        sx2 = sx1 + tw

        # 获取未探索区域（安全索引）
        region_mask = self.explored_mask[y1c:y2c, x1c:x2c]
        unexplored = region_mask == 0

        if np.any(unexplored):
            for c in range(3):
                dst = self.canvas[y1c:y2c, x1c:x2c, c]
                src = minimap[sy1:sy2, sx1:sx2, c]
                dst[unexplored] = src[unexplored]
                self.canvas[y1c:y2c, x1c:x2c, c] = dst
            self.explored_mask[y1c:y2c, x1c:x2c][unexplored] = 255

        return True

    def stitch_smart(self, minimap, diff_threshold=30, blend_weight=0.3):
        """
        智能混合拼接模式（基于权重融合消除裂纹）：
        1. 未探索区域：直接填充新内容
        2. 已探索区域：加权融合新旧像素（消除累积误差导致的裂纹）
        3. 差异大的区域：直接更新（迷雾揭开）

        diff_threshold: 像素差异阈值，超过此值认为是新内容需要更新
        blend_weight: 已探索区域的新像素混合权重（0-1），越大越偏向新像素
        """
        h, w = minimap.shape[:2]

        # 使用整数并裁剪到画布边界
        y1 = int(round(self.canvas_y))
        x1 = int(round(self.canvas_x))
        y2 = y1 + h
        x2 = x1 + w

        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(self.canvas_size, y2)
        x2c = min(self.canvas_size, x2)

        th = y2c - y1c
        tw = x2c - x1c
        if th <= 0 or tw <= 0:
            print(f"  ⚠️ 警告: 超出画布范围 ({x1},{y1})-({x2},{y2})")
            return False

        # 获取当前区域（安全切片）
        canvas_region = self.canvas[y1c:y2c, x1c:x2c]
        region_mask = self.explored_mask[y1c:y2c, x1c:x2c]
        # 对应源图切片
        sy1 = max(0, - (y1 - y1c))
        sx1 = max(0, - (x1 - x1c))
        sy2 = sy1 + th
        sx2 = sx1 + tw
        minimap = minimap[sy1:sy2, sx1:sx2]

        # 1. 未探索区域：直接填充
        unexplored = region_mask == 0
        if np.any(unexplored):
            for c in range(3):
                canvas_region[:, :, c][unexplored] = minimap[:, :, c][unexplored]
            self.explored_mask[y1c:y2c, x1c:x2c][unexplored] = 255

        # 2. 已探索区域：智能融合
        explored = region_mask > 0
        if np.any(explored):
            # 计算差异
            diff = np.abs(canvas_region.astype(np.int16) - minimap.astype(np.int16))
            max_diff = np.max(diff, axis=2)

            # 差异大的区域直接更新（迷雾揭开等）
            significant_change = (max_diff > diff_threshold) & explored
            if np.any(significant_change):
                for c in range(3):
                    canvas_region[:, :, c][significant_change] = minimap[:, :, c][significant_change]

            # 差异小的已探索区域：加权融合（消除裂纹）
            # 使用较小权重混合新像素，逐渐平滑累积误差
            blend_region = explored & (~significant_change)
            if np.any(blend_region):
                for c in range(3):
                    old_val = canvas_region[:, :, c][blend_region].astype(np.float32)
                    new_val = minimap[:, :, c][blend_region].astype(np.float32)
                    # 加权平均：保留大部分旧值，混入少量新值
                    blended = (1 - blend_weight) * old_val + blend_weight * new_val
                    canvas_region[:, :, c][blend_region] = blended.astype(np.uint8)

        return True

    def stitch_seam_blend(self, minimap, seam_width=15):
        """
        接缝融合拼接模式：
        在新旧区域的交界处使用距离加权融合，彻底消除裂纹

        seam_width: 接缝融合宽度（像素）
        """
        h, w = minimap.shape[:2]
        # 使用四舍五入到整数位置
        y1 = int(round(self.canvas_y))
        y2 = y1 + h
        x1 = int(round(self.canvas_x))
        x2 = x1 + w

        # 边界检查
        if y1 < 0 or x1 < 0 or y2 > self.canvas_size or x2 > self.canvas_size:
            print(f"  ⚠️ 警告: 超出画布范围 ({x1},{y1})-({x2},{y2})")
            return False

        # 获取当前区域
        canvas_region = self.canvas[y1:y2, x1:x2]
        region_mask = self.explored_mask[y1:y2, x1:x2]

        # 未探索区域
        unexplored = region_mask == 0
        # 已探索区域
        explored = region_mask > 0

        if not np.any(unexplored):
            # 全部已探索，不更新（往回走时保护已有内容）
            return True

        if not np.any(explored):
            # 全部未探索，直接填充
            canvas_region[:] = minimap
            self.explored_mask[y1:y2, x1:x2] = 255
            return True

        # 计算距离图：每个像素到"未探索区域"的距离
        dist_to_unexplored = cv2.distanceTransform(
            (~unexplored).astype(np.uint8) * 255,
            cv2.DIST_L2, 5
        )

        # 创建融合权重图
        # alpha: 0=完全用新像素，1=完全用旧像素
        alpha = np.clip(dist_to_unexplored / (seam_width + 1e-6), 0, 1)

        # 在完全未探索区域，alpha=0（用新像素）
        alpha[unexplored] = 0
        # 在远离接缝的已探索区域，alpha=1（用旧像素）
        alpha[dist_to_unexplored > seam_width] = 1

        # 执行融合
        for c in range(3):
            old_val = canvas_region[:, :, c].astype(np.float32)
            new_val = minimap[:, :, c].astype(np.float32)
            blended = alpha * old_val + (1 - alpha) * new_val
            canvas_region[:, :, c] = blended.astype(np.uint8)

        # 更新探索遮罩
        self.explored_mask[y1:y2, x1:x2][unexplored] = 255

        return True

    def stitch_accumulate(self, minimap):
        """
        累积平均拼接算法（核心方法）

        三种策略处理不同区域：
        1. 未探索区域 -> 直接填充
        2. 更清晰内容 -> 直接替换（边界动态扩展）
        3. 显著变化 -> 高权重累积（迷雾揭开）
        4. 微小变化 -> 正常累积平均（消除裂纹）

        Args:
            minimap: 当前帧小地图图像

        Returns:
            bool: 拼接是否成功
        """
        h, w = minimap.shape[:2]
        y1 = int(round(self.canvas_y))
        y2 = y1 + h
        x1 = int(round(self.canvas_x))
        x2 = x1 + w

        # 边界检查
        if y1 < 0 or x1 < 0 or y2 > self.canvas_size or x2 > self.canvas_size:
            print(f"  ⚠️ 警告: 超出画布范围 ({x1},{y1})-({x2},{y2})")
            return False

        # 获取当前区域（视图引用，直接修改会更新原始数组）
        canvas_region = self.canvas[y1:y2, x1:x2]
        region_mask = self.explored_mask[y1:y2, x1:x2]
        region_count = self.canvas_count[y1:y2, x1:x2]
        region_accum = self.canvas_accum[y1:y2, x1:x2]

        # === 策略1: 未探索区域直接填充 ===
        unexplored = region_mask == 0
        if np.any(unexplored):
            minimap_f64 = minimap.astype(np.float64)
            for c in range(3):
                canvas_region[:, :, c][unexplored] = minimap[:, :, c][unexplored]
                region_accum[:, :, c][unexplored] = minimap_f64[:, :, c][unexplored]
            region_count[unexplored] = 1
            self.explored_mask[y1:y2, x1:x2][unexplored] = 255

        # === 策略2-4: 已探索区域 ===
        explored = region_mask > 0
        if not np.any(explored):
            return True

        # 计算像素差异（BGR 三通道最大差异）
        diff = np.abs(canvas_region.astype(np.int16) - minimap.astype(np.int16))
        max_diff = np.max(diff, axis=2)

        # 转换到 HSV 比较清晰度
        canvas_hsv = cv2.cvtColor(canvas_region, cv2.COLOR_BGR2HSV)
        minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # 饱和度 + 亮度综合判断清晰度
        old_clarity = canvas_hsv[:, :, 1].astype(np.int16) + canvas_hsv[:, :, 2].astype(np.int16)
        new_clarity = minimap_hsv[:, :, 1].astype(np.int16) + minimap_hsv[:, :, 2].astype(np.int16)

        # 分类已探索区域的像素
        clearer = (new_clarity > old_clarity + self.clarity_threshold) & explored
        significant = (max_diff > self.diff_threshold) & explored & (~clearer)
        minor = explored & (~significant) & (~clearer)

        minimap_f64 = minimap.astype(np.float64)

        # === 策略2: 更清晰内容直接替换（边界扩展） ===
        if np.any(clearer):
            for c in range(3):
                canvas_region[:, :, c][clearer] = minimap[:, :, c][clearer]
                region_accum[:, :, c][clearer] = minimap_f64[:, :, c][clearer]
            region_count[clearer] = 1

        # === 策略3: 显著变化高权重累积（迷雾揭开） ===
        if np.any(significant):
            weight = self.significant_weight
            for c in range(3):
                region_accum[:, :, c][significant] += minimap_f64[:, :, c][significant] * weight
            region_count[significant] += weight
            # 更新显示
            for c in range(3):
                avg_val = region_accum[:, :, c][significant] / region_count[significant]
                canvas_region[:, :, c][significant] = avg_val.astype(np.uint8)

        # === 策略4: 微小变化正常累积（消除裂纹） ===
        if np.any(minor):
            for c in range(3):
                region_accum[:, :, c][minor] += minimap_f64[:, :, c][minor]
            region_count[minor] += 1
            # 更新显示
            for c in range(3):
                avg_val = region_accum[:, :, c][minor] / region_count[minor]
                canvas_region[:, :, c][minor] = avg_val.astype(np.uint8)

        return True

    def _create_feather_mask(self, h, w, feather_width):
        """
        创建边缘羽化遮罩
        中心为1（保留旧值），边缘为0（使用新值），过渡区域渐变
        """
        mask = np.ones((h, w), dtype=np.float32)

        if feather_width <= 0:
            return mask

        # 上下边缘羽化
        for i in range(min(feather_width, h // 2)):
            alpha = i / feather_width
            mask[i, :] = min(mask[i, :].min(), alpha)
            mask[h - 1 - i, :] = min(mask[h - 1 - i, :].min(), alpha)

        # 左右边缘羽化
        for j in range(min(feather_width, w // 2)):
            alpha = j / feather_width
            mask[:, j] = np.minimum(mask[:, j], alpha)
            mask[:, w - 1 - j] = np.minimum(mask[:, w - 1 - j], alpha)

        return mask

    def run(self):
        """主循环"""
        # 截取初始帧
        minimap = self.capture_minimap()
        h, w = minimap.shape[:2]

        print(f"\n小地图尺寸: {w} x {h}")
        print(f"初始位置: ({self.canvas_x}, {self.canvas_y})")

        # 放置初始帧到画布中心
        cx, cy = int(self.canvas_x), int(self.canvas_y)
        self.canvas[cy:cy+h, cx:cx+w] = minimap
        # 初始化累积画布
        self.canvas_accum[cy:cy+h, cx:cx+w] = minimap.astype(np.float64)
        self.canvas_count[cy:cy+h, cx:cx+w] = 1
        self.explored_mask[cy:cy+h, cx:cx+w] = 255
        self.last_minimap = minimap.copy()


        # cv2.namedWindow("minimap_stitch_smart", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('minimap_stitch_smart', 800, 800)

        last_report_time = time.time()

        try:
            while True:
                # 截取新帧
                minimap_new = self.capture_minimap()
                self.frame_count += 1

                # 检测移动
                has_moved, dx, dy, confidence = self.detect_movement(
                    self.last_minimap,
                    minimap_new
                )

                if has_moved:
                    # 更新位置（浮点精度）
                    self.canvas_x -= dx
                    self.canvas_y -= dy

                    # 拼接（累积平均模式）
                    success = self.stitch(minimap_new)

                    if success:
                        self.stitch_count += 1

                    # 更新上一帧
                    self.last_minimap = minimap_new.copy()

                # 裁剪并显示结果
                gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                mask = gray_canvas != 200
                coords = np.argwhere(mask)

                if len(coords) > 0:
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0)
                    cropped = self.canvas[y0:y1 + 1, x0:x1 + 1]

                    # 更新共享结果（使用 Queue 进行进程间通信）
                    if result_queue is not None:
                        # 清空队列保持最新
                        while not result_queue.empty():
                            try:
                                result_queue.get_nowait()
                            except:
                                break
                        # 放入最新结果
                        result_queue.put(cropped.copy())

                    # # 显示
                    # cv2.imshow("Minimap Stitching Result", cropped)

                    # # 自动保存
                    # cv2.imwrite("final_smart_stitch_cropped.png", cropped)

                # 处理按键
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("\n用户退出")
                    break
                elif key == ord('s'):
                    if len(coords) > 0:
                        timestamp = int(time.time())
                        filename = f"stitch_save_{timestamp}.png"
                        cv2.imwrite(filename, cropped)
                        print(f"  已保存: {filename}")

                # 降低 CPU 占用
                time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n中断退出")
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 输出统计信息
            print("\n" + "=" * 60)
            print("运行统计")
            print("=" * 60)
            print(f"  总帧数: {self.frame_count}")
            print(f"  拼接次数: {self.stitch_count}")
            if self.frame_count > 0:
                print(f"  拼接率: {self.stitch_count / self.frame_count * 100:.1f}%")
            print("=" * 60)
            cv2.destroyAllWindows()


def stitch_worker_process(queue, x1, y1, x2, y2, name, **kwargs):
    """
    拼接工作进程函数

    Args:
        queue: 用于传递结果的队列
        x1, y1, x2, y2: 小地图截图区域
        name: 游戏窗口标题
        **kwargs: 其他参数传递给 SmartMinimapStitcher
    """
    global result_queue
    result_queue = queue

    stitcher = SmartMinimapStitcher(
        x1=x1, y1=y1, x2=x2, y2=y2,
        name=name,
        **kwargs
    )
    stitcher.run()


if __name__ == '__main__':
    """
    主程序入口
    
    使用示例:
        默认配置（POE2）:
        python stitch_new.py
        
        自定义参数:
        stitcher = SmartMinimapStitcher(
            x1=1100, y1=33, x2=1261, y2=176,
            name="Path of Exile 2",
            canvas_size=3000,
            movement_threshold=3.0,
            diff_threshold=25,
            clarity_threshold=10
        )
    """
    stitcher = SmartMinimapStitcher(
        x1=1100, y1=33,
        x2=1261, y2=176,
        name="Path of Exile 2",
        canvas_size=3000,
        movement_threshold=3.0,
        confidence_threshold=0.25,
        diff_threshold=25,
        clarity_threshold=10,
        significant_weight=5.0
    )

    try:
        stitcher.run()
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束")
        cv2.destroyAllWindows()
