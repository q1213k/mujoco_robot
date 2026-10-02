import math                   # 标准数学库：log、gamma（计算 RRT* 自适应半径与单位球体积）
import numpy as np            # 数值计算库：数组、随机采样、距离计算（RRT* 手写实现依赖）
import mujoco                # MuJoCo 物理引擎：把候选关节角写入 qpos 后做碰撞检测


class RRTStarPlanner:  # 定义纯 Python 实现的 RRT* 路径规划器类
    """ 参数：
        model : MuJoCo 模型（读取关节限位、做碰撞检测）
        data  : MuJoCo 数据（写入关节角并步进仿真）
        n_dof : 参与规划的机械臂关节数（默认 6）
    """

    def __init__(self, model, data, n_dof=6):  # 初始化规划器，保存模型、数据并缓存关节限位
        self.model = model                                   # 保存 MuJoCo 模型引用
        self.data = data                                     # 保存 MuJoCo 数据引用
        self.n_dof = n_dof                                   # 参与规划的关节数
        self.low = self.model.jnt_range[:n_dof, 0].copy()    # 各关节下限（用于采样）
        self.high = self.model.jnt_range[:n_dof, 1].copy()   # 各关节上限（用于采样）
        # 自适应邻域半径系数 γ = 2·(1+1/d)^(1/d)·(μ_free/ζ_d)^(1/d)（Karaman & Frazzoli）  # RRT* 标准半径公式
        zeta = math.pi ** (n_dof / 2.0) / math.gamma(n_dof / 2.0 + 1.0)  # d 维单位球体积
        mu = float(np.prod(self.high - self.low))                        # 构型空间（关节限位盒）体积
        self.gamma = 2.0 * (1.0 + 1.0 / n_dof) ** (1.0 / n_dof) * (mu / zeta) ** (1.0 / n_dof)  # 半径系数

    def _collision_free(self, q):  # 单点构型碰撞检测：用 MuJoCo 碰撞检测（data.ncon == 0）判断构型是否合法
        self.data.qpos[:self.n_dof] = q                      # 把构型写入前 n_dof 个关节
        mujoco.mj_step(self.model, self.data)                # 步进仿真以刷新接触信息
        return self.data.ncon == 0                           # 接触对为零返回True即合法

    def _edge_free(self, a, b, n=8):  # 边碰撞检测：沿 a→b 线段插值采样 n 段
        for t in np.linspace(0.0, 1.0, n + 1):               # 生成 0~1 的n+1个等比例插值数组
            if not self._collision_free(a + t * (b - a)):    # 任一插值点碰撞则边不合法
                return False                                 # 提前返回失败
        return True                                          # 整条边均无碰撞
    
    def _steer(self, a, b, step):  # 步进方法，从 a 朝 b 方向前进 step 长度
        d = b - a                                            # 方向向量
        dist = np.linalg.norm(d)                             # 两点距离
        if dist <= step:                                     # 已足够近，直接一步落到 b
            return b                                         # 返回目标点
        return a + d / dist * step                           # 否则沿方向前进 step

    def _nearest(self, nodes, q):  # 找距 q 最近的节点下标
        return int(np.argmin([np.linalg.norm(n - q) for n in nodes]))  # 返回与q距离最小的那个节点的索引

    def _near(self, nodes, q, radius):  # 找半径 radius 内所有邻居下标
        return [i for i, n in enumerate(nodes) if np.linalg.norm(n - q) <= radius]  # 找出所有与q距离不超过radius的现有节点（方括号创建列表）

    def plan(self, start_q, goal_q, max_iter=2000, step=0.05, goal_rate=0.1):  # 规划主入口
        """规划从 start_q 到 goal_q 的无碰撞路径（RRT*）。 
        参数：
            start_q   : 起始关节角（长度 n_dof 的序列）
            goal_q    : 目标关节角（长度 n_dof 的序列）
            max_iter  : 最大迭代次数，达到后停止
            step      : 单次扩展步长 (rad)
            goal_rate : 直接采样目标的概率（goal bias）

        邻域半径用 RRT* 标准自适应公式 γ·(log n / n)^(1/d) 随树规模收缩， 
        返回路径点列表（起点到终点，每个点 n_dof 个关节角）
        """
        start_q = np.asarray(start_q, dtype=float)[:self.n_dof]  # 起始构型转数组
        goal_q = np.asarray(goal_q, dtype=float)[:self.n_dof]    # 目标构型转数组

        nodes = [start_q.copy()]  # 树节点列表，首节点为起点
        parent = [-1]             # 每个节点的父节点下标，起点无父节点
        cost = [0.0]              # 从起点累积到各节点的路径代价
        goal_idx = -1             # 目标节点下标（-1 表示尚未连入）

        for _ in range(max_iter):                              # 迭代扩展直到预算耗尽
            if np.random.rand() < goal_rate:                   # 按概率直接采样目标
                sample = goal_q                               # 采样目标构型
            else:                                              # 否则均匀随机采样
                sample = np.random.uniform(self.low, self.high)  # 限位内随机构型

            nearest = self._nearest(nodes, sample)             # 在节点列表中查找与采样点最近的节点的索引
            new = self._steer(nodes[nearest], sample, step)    # 朝样本扩展一步得到新节点
            if not self._edge_free(nodes[nearest], new):       # 扩展边碰撞则放弃本次
                continue                                       # 进入下一轮迭代重新采样

            n = len(nodes)                                     # 当前树节点数
            radius = self.gamma * (math.log(n) / n) ** (1.0 / self.n_dof)  # 自适应邻域半径 γ·(log n / n)^(1/d)
            neighbors = self._near(nodes, new, radius)         # 查找新节点半径内的所有节点

            # 择优父节点：选连接后累积代价最小的邻居作为父节点  # RRT* 关键：代价优先
            best_parent = nearest                              # 默认父节点为最近节点
            best_cost = cost[nearest] + np.linalg.norm(new - nodes[nearest])  # 从起点到 new 的累积路径代价
            for i in neighbors:                                # 遍历邻居找更优父节点
                c = cost[i] + np.linalg.norm(new - nodes[i])   # 经邻居连接的候选代价
                if c < best_cost and self._edge_free(nodes[i], new):  # 代价更小且边无碰撞
                    best_parent, best_cost = i, c              # 更新最优父节点与代价

            new_idx = len(nodes)                               # 新节点下标
            nodes.append(new)                                  # 加入树
            parent.append(best_parent)                         # 记录父节点
            cost.append(best_cost)                             # 记录累积代价

            # 重连：若经新节点可降低邻居代价，则把邻居父节点改为新节点
            '''因为采样点是随机生成不带方向感的，所以新节点的位置可能会比邻居更靠近起点，此时邻居经过新节点到起点的代价可能会更小，
            因此需要检查邻居是否可以通过新节点重连以降低代价。'''

            for i in neighbors:                                # 遍历邻居
                c = best_cost + np.linalg.norm(nodes[i] - new)  # 经新节点到达邻居的代价
                if c < cost[i] and self._edge_free(new, nodes[i]):  # 更优且边无碰撞
                    parent[i], cost[i] = new_idx, c            # 重连并更新代价

            # 目标连接：新节点已能一步抵达目标则直接连入并结束
            if np.linalg.norm(new - goal_q) <= step and self._edge_free(new, goal_q):  # 可达目标
                goal_idx = len(nodes)                          # 更新目标节点下标
                nodes.append(goal_q.copy())                    # 目标作为末节点加入
                parent.append(new_idx)                         # 目标父节点为新节点
                print(f"RRT* 找到路径，迭代 {_+1} 次，节点数 {len(nodes)}，邻域半径 {radius:.4f}")  # 打印调试信息
                break                                          # 找到解，停止迭代

        # 路径回溯：从目标节点（或距目标最近节点）沿父节点回到起点
        if goal_idx < 0:                                       # 未直接连入目标
            goal_idx = self._nearest(nodes, goal_q)            # 退化为距目标最近的节点
            print(f"RRT* 未直接连入目标，退化为距目标最近的节点 {goal_idx}，距离 {np.linalg.norm(nodes[goal_idx] - goal_q):.4f}")  # 打印调试信息
        path = []                                              # 路径点列表
        idx = goal_idx                                         # 从末端节点开始
        while idx >= 0: # 将当前节点存入路径并获取其父节点的索引重复存入
            path.append(nodes[idx].tolist())                   # 收集当前节点
            idx = parent[idx]                                  # 跳到父节点
        return path[::-1]                                      # 反转为起点到终点的顺序



    










# 没有经过优化，做路径规划不好用，只能用于对比学习------------------------------------------------------------------------------
class PlainRRTPlanner:  # 定义纯 Python 实现的普通 RRT 路径规划器类
    def __init__(self, model, data, n_dof=6):  # 初始化规划器，保存模型、数据并缓存关节限位
        self.model = model                                   # 保存 MuJoCo 模型引用
        self.data = data                                     # 保存 MuJoCo 数据引用
        self.n_dof = n_dof                                   # 参与规划的关节数
        self.low = self.model.jnt_range[:n_dof, 0].copy()    # 各关节下限（用于采样）
        self.high = self.model.jnt_range[:n_dof, 1].copy()   # 各关节上限（用于采样）

    def _collision_free(self, q):  # 单点构型碰撞检测：写入关节角后步进，判断是否无接触
        self.data.qpos[:self.n_dof] = q                      # 把构型写入前 n_dof 个关节
        mujoco.mj_step(self.model, self.data)                # 步进仿真以刷新接触信息
        return self.data.ncon == 0                           # 接触对为零即合法

    def _edge_free(self, a, b, n=8):  # 边碰撞检测：沿 a→b 线段插值采样 n 段
        for t in np.linspace(0.0, 1.0, n + 1):               # 生成 0~1 的插值比例
            if not self._collision_free(a + t * (b - a)):    # 任一插值点碰撞则边不合法
                return False                                 # 提前返回失败
        return True                                          # 整条边均无碰撞

    def _steer(self, a, b, step):  # 从 a 朝 b 方向前进 step 长度
        d = b - a                                            # 方向向量
        dist = np.linalg.norm(d)                             # 两点距离
        if dist <= step:                                     # 已足够近，直接落到 b
            return b                                         # 返回目标点
        return a + d / dist * step                           # 否则沿方向前进 step

    def _nearest(self, nodes, q):  # 找距 q 最近的节点下标
        return int(np.argmin([np.linalg.norm(n - q) for n in nodes]))  # 欧氏距离最小者

    def plan(self, start_q, goal_q, max_iter=2000, step=0.05, goal_rate=0.1):  # 规划主入口
        start_q = np.asarray(start_q, dtype=float)[:self.n_dof]  # 起始构型转数组
        goal_q = np.asarray(goal_q, dtype=float)[:self.n_dof]    # 目标构型转数组

        nodes = [start_q.copy()]  # 树节点列表，首节点为起点
        parent = [-1]             # 每个节点的父节点下标，起点无父节点
        goal_idx = -1             # 目标节点下标（-1 表示尚未连入）

        for _ in range(max_iter):                              # 迭代扩展直到预算耗尽
            if np.random.rand() < goal_rate:                   # 按概率直接采样目标
                sample = goal_q                               # 采样目标构型
            else:                                              # 否则均匀随机采样
                sample = np.random.uniform(self.low, self.high)  # 限位内随机构型

            nearest = self._nearest(nodes, sample)             # 找最近节点
            new = self._steer(nodes[nearest], sample, step)    # 朝样本扩展一步
            if not self._edge_free(nodes[nearest], new):       # 扩展边碰撞则放弃本次
                continue                                       # 进入下一轮迭代

            new_idx = len(nodes)                               # 新节点下标
            nodes.append(new)                                  # 加入树
            parent.append(nearest)                             # 直接连到最近节点

            # 目标连接：新节点已能一步抵达目标则直接连入并结束  # 提前收敛
            if np.linalg.norm(new - goal_q) <= step and self._edge_free(new, goal_q):  # 可达目标
                goal_idx = len(nodes)                          # 目标节点下标
                nodes.append(goal_q.copy())                    # 目标作为末节点加入
                parent.append(new_idx)                         # 目标父节点为新节点
                break                                          # 找到解，停止迭代

        # 路径回溯：从目标节点（或距目标最近节点）沿父节点回到起点  # 反向回溯
        if goal_idx < 0:                                       # 未直接连入目标
            goal_idx = self._nearest(nodes, goal_q)            # 退化为距目标最近的节点
        path = []                                              # 路径点列表
        idx = goal_idx                                         # 从末端节点开始
        while idx >= 0:                                        # 沿父节点链回溯到起点
            path.append(nodes[idx].tolist())                   # 收集当前节点
            idx = parent[idx]                                  # 跳到父节点
        return path[::-1]                                      # 反转为起点到终点的顺序
    
