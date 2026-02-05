"""
优化的快速探索算法 - 减少回头路
基于区域优先和距离-收益平衡策略
"""

import cv2
import numpy as np
import math
import heapq


class OptimizedFastExplorer:
    """优化的快速探索器 - 减少回头路"""

    def __init__(self, map_path, vision_radius=6, target_coverage=85.0):
        """
        初始化探索器

        Args:
            map_path: 地图图片路径
            vision_radius: 视野半径（像素）
            target_coverage: 目标覆盖率（百分比）
        """
        # 加载地图
        self.original_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.original_map is None:
            raise ValueError(f"无法加载地图: {map_path}")

        self.height, self.width = self.original_map.shape
        print(f"地图加载成功: {self.width}x{self.height}")
        print(f"视野半径: {vision_radius}像素")
        print(f"目标覆盖率: {target_coverage}%")

        # 二值化地图
        _, self.walkable_map = cv2.threshold(self.original_map, 127, 255, cv2.THRESH_BINARY)

        self.vision_radius = vision_radius
        self.target_coverage = target_coverage
        self.path = []
        self.key_points = []
        self.coverage_map = np.zeros((self.height, self.width), dtype=np.uint8)

        self.total_walkable = np.sum(self.walkable_map == 255)
        print(f"白色路径像素数: {self.total_walkable}")

    def is_walkable(self, x, y):
        """检查位置是否可行走"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.walkable_map[y, x] == 255
        return False

    def mark_vision_coverage(self, x, y):
        """标记视野覆盖"""
        y_min = max(0, y - self.vision_radius)
        y_max = min(self.height, y + self.vision_radius + 1)
        x_min = max(0, x - self.vision_radius)
        x_max = min(self.width, x + self.vision_radius + 1)

        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                dist = math.sqrt((px - x) ** 2 + (py - y) ** 2)
                if dist <= self.vision_radius and self.walkable_map[py, px] == 255:
                    self.coverage_map[py, px] = 255

    def get_coverage_score(self, x, y):
        """计算覆盖收益"""
        if not self.is_walkable(x, y):
            return 0

        y_min = max(0, y - self.vision_radius)
        y_max = min(self.height, y + self.vision_radius + 1)
        x_min = max(0, x - self.vision_radius)
        x_max = min(self.width, x + self.vision_radius + 1)

        new_coverage = 0
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                dist = math.sqrt((px - x) ** 2 + (py - y) ** 2)
                if dist <= self.vision_radius:
                    if self.walkable_map[py, px] == 255 and self.coverage_map[py, px] == 0:
                        new_coverage += 1

        return new_coverage

    def estimate_path_distance(self, start, goal):
        """
        估算路径距离（曼哈顿距离）
        比A*快，用于快速评估
        """
        return abs(goal[0] - start[0]) + abs(goal[1] - start[1])

    def find_path_astar(self, start, goal, max_distance=200):
        """A*寻路"""
        if not self.is_walkable(start[0], start[1]) or not self.is_walkable(goal[0], goal[1]):
            return []

        # 距离太远直接放弃
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) > max_distance:
            return []

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        max_iterations = 10000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_walkable(neighbor[0], neighbor[1]):
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def find_best_nearby_point(self, current_pos, search_radius=50, distance_weight=0.3):
        """
        找最佳附近点 - 考虑距离和收益的平衡

        Args:
            current_pos: 当前位置
            search_radius: 搜索半径
            distance_weight: 距离权重（0-1，越大越重视距离）

        Returns:
            (best_pos, score, distance)
        """
        cx, cy = current_pos
        best_pos = None
        best_score = -float('inf')
        best_distance = 0

        # 使用较小的步长进行精细搜索
        step = max(self.vision_radius, 5)

        y_min = max(0, cy - search_radius)
        y_max = min(self.height, cy + search_radius + 1)
        x_min = max(0, cx - search_radius)
        x_max = min(self.width, cx + search_radius + 1)

        candidates = []

        for py in range(y_min, y_max, step):
            for px in range(x_min, x_max, step):
                if not self.is_walkable(px, py):
                    continue

                # 计算覆盖收益
                coverage_score = self.get_coverage_score(px, py)
                if coverage_score == 0:
                    continue

                # 计算距离（归一化到0-1）
                distance = self.estimate_path_distance(current_pos, (px, py))
                normalized_distance = distance / (search_radius * 2)

                # 综合评分：收益 - 距离惩罚
                # 距离越近越好，收益越大越好
                combined_score = coverage_score * (1 - distance_weight) - normalized_distance * 100 * distance_weight

                candidates.append((combined_score, coverage_score, distance, (px, py)))

        # 选择综合评分最高的点
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_score, coverage_score, best_distance, best_pos = candidates[0]
            return best_pos, coverage_score, best_distance

        return None, 0, 0

    def find_best_point_in_region(self, current_pos, min_radius=30, max_radius=80, distance_weight=0.4):
        """
        在区域内找最佳点 - 渐进式搜索
        先在近处找，近处没有好的再扩大范围
        """
        # 先在近处找
        pos, score, dist = self.find_best_nearby_point(current_pos, min_radius, distance_weight)

        if pos and score >= max(10, int(self.vision_radius * 1.5)):
            return pos, score, dist

        # 近处不理想，扩大范围
        pos, score, dist = self.find_best_nearby_point(current_pos, max_radius, distance_weight * 0.8)

        return pos, score, dist

    def optimized_explore(self):
        """
        优化的探索算法 - 减少回头路
        """
        # 找起始位置
        start_x, start_y = self.width // 2, self.height // 2

        if not self.is_walkable(start_x, start_y):
            for r in range(1, max(self.width, self.height)):
                for angle in range(0, 360, 15):
                    rad = math.radians(angle)
                    x = int(start_x + r * math.cos(rad))
                    y = int(start_y + r * math.sin(rad))
                    if self.is_walkable(x, y):
                        start_x, start_y = x, y
                        break
                if self.is_walkable(start_x, start_y):
                    break

        print(f"起始位置: ({start_x}, {start_y})")

        current_pos = (start_x, start_y)
        self.mark_vision_coverage(current_pos[0], current_pos[1])
        self.key_points.append(current_pos)
        self.path.append(current_pos)

        iteration = 0
        max_iterations = 500

        # 动态调整距离权重：开始时重视距离（避免跳跃），后期降低距离权重（覆盖更多区域）

        print("\n开始优化探索...")

        while iteration < max_iterations:
            iteration += 1

            covered = np.sum(self.coverage_map == 255)
            coverage_rate = covered / self.total_walkable * 100 if self.total_walkable > 0 else 0

            if iteration % 10 == 0 or iteration == 1:
                print(f"迭代 {iteration}: 覆盖率 {coverage_rate:.1f}%, 停留点 {len(self.key_points)}, 总步数 {len(self.path)}")

            # 达到目标
            if coverage_rate >= self.target_coverage:
                print(f"达到目标覆盖率 {self.target_coverage}%，探索完成！")
                break

            # 根据覆盖率动态调整距离权重
            # 开始时（覆盖率低）：更重视距离，避免跳跃
            # 后期时（覆盖率高）：降低距离权重，找更多区域
            if coverage_rate < 40:
                distance_weight = 0.5  # 前期：非常重视距离
            elif coverage_rate < 70:
                distance_weight = 0.3  # 中期：平衡
            else:
                distance_weight = 0.2  # 后期：更重视覆盖

            # 在当前区域内找最佳点（渐进式）
            next_pos, score, distance = self.find_best_point_in_region(
                current_pos,
                min_radius=self.vision_radius * 4,
                max_radius=self.vision_radius * 10,
                distance_weight=distance_weight
            )

            if next_pos is None or score < max(3, self.vision_radius // 2):
                # 当前区域探索完毕，寻找远处未探索区域
                # 此时降低距离权重，因为必须跳跃了
                next_pos, score, distance = self.find_best_nearby_point(
                    current_pos,
                    search_radius=max(100, self.vision_radius * 15),
                    distance_weight=0.1  # 降低距离权重
                )

            if next_pos is None or score == 0:
                print(f"迭代 {iteration}: 没有更多高价值区域")
                break

            # 寻路
            max_path_distance = 150 if distance < 100 else 250
            path_to_next = self.find_path_astar(current_pos, next_pos, max_distance=max_path_distance)

            if not path_to_next:
                # 无法到达，尝试更近的点
                next_pos, score, distance = self.find_best_nearby_point(
                    current_pos,
                    search_radius=self.vision_radius * 5,
                    distance_weight=0.5
                )

                if next_pos:
                    path_to_next = self.find_path_astar(current_pos, next_pos, max_distance=100)

                if not path_to_next:
                    print(f"  无法找到可达的探索点")
                    break

            # 移动
            for pos in path_to_next[1:]:
                self.path.append(pos)

            current_pos = next_pos
            self.mark_vision_coverage(current_pos[0], current_pos[1])
            self.key_points.append(current_pos)

        # 最终统计
        covered = np.sum(self.coverage_map == 255)
        coverage_rate = covered / self.total_walkable * 100 if self.total_walkable > 0 else 0

        print(f"\n=== 优化探索完成 ===")
        print(f"总白色路径: {self.total_walkable}")
        print(f"已覆盖: {covered}")
        print(f"覆盖率: {coverage_rate:.2f}%")
        print(f"停留点数: {len(self.key_points)}")
        print(f"总路径点数: {len(self.path)}")

        # 计算平均移动距离（衡量回头路）
        total_distance = 0
        for i in range(1, len(self.key_points)):
            prev = self.key_points[i-1]
            curr = self.key_points[i]
            total_distance += abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])

        avg_distance = total_distance / (len(self.key_points) - 1) if len(self.key_points) > 1 else 0

        print(f"平均停留点间距: {avg_distance:.1f}像素")
        print(f"效率: 每个停留点覆盖 {covered/len(self.key_points):.1f} 像素" if self.key_points else "")

        return self.path

    def visualize_result(self, output_path='optimized_result.png'):
        """可视化"""
        result = cv2.cvtColor(self.original_map, cv2.COLOR_GRAY2BGR)

        # 覆盖区域
        green_overlay = np.zeros_like(result)
        green_overlay[:, :] = [100, 255, 100]
        coverage_mask = self.coverage_map > 0
        result = np.where(coverage_mask[:, :, np.newaxis],
                         (result * 0.95 + green_overlay * 0.05).astype(np.uint8),
                         result)

        # 路径
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                pt1 = self.path[i]
                pt2 = self.path[i + 1]
                cv2.line(result, pt1, pt2, (0, 255, 255), 1)

        # 停留点
        for x, y in self.key_points:
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)

        # 起终点
        if self.key_points:
            start = self.key_points[0]
            end = self.key_points[-1]

            cv2.circle(result, start, 8, (0, 255, 0), -1)
            cv2.circle(result, start, 8, (255, 255, 255), 2)
            cv2.putText(result, "START", (start[0] + 12, start[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.circle(result, end, 8, (0, 0, 255), -1)
            cv2.circle(result, end, 8, (255, 255, 255), 2)
            cv2.putText(result, "END", (end[0] + 12, end[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite(output_path, result)
        print(f"\n可视化结果: {output_path}")

        print("\n图例:")
        print("  - 淡绿色: 已覆盖区域")
        print("  - 黄色细线: 移动路径")
        print("  - 红色圆点: 停留点")
        print("  - 绿色: 起点")
        print("  - 红色: 终点")

        return result

    def save_path(self, output_path='optimized_path.txt'):
        """保存路径"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 优化的快速探索路径（减少回头路）\n")
            f.write(f"# 视野半径: {self.vision_radius}像素\n")
            f.write(f"# 目标覆盖率: {self.target_coverage}%\n")
            f.write(f"# 停留点数: {len(self.key_points)}\n")
            f.write(f"# 总路径点数: {len(self.path)}\n")
            f.write(f"# 格式: 序号,x,y\n")
            for i, (x, y) in enumerate(self.path):
                f.write(f"{i},{x},{y}\n")
        print(f"路径数据: {output_path}")


def main():
    print("=" * 80)
    print("优化的快速探索 - 减少回头路")
    print("=" * 80)

    map_path = 'route.png'
    vision_radius = 6
    target_coverage = 85.0

    try:
        explorer = OptimizedFastExplorer(map_path,
                                        vision_radius=vision_radius,
                                        target_coverage=target_coverage)

        path = explorer.optimized_explore()

        explorer.visualize_result('optimized_result.png')
        explorer.save_path('optimized_path.txt')

        print("\n✅ 探索完成！")
        print("文件:")
        print("  - optimized_result.png")
        print("  - optimized_path.txt")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

