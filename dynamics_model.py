"""ARX X5a 机械臂 —— 递归牛顿-欧拉（RNEA）动力学建模。

以 arx_x5a/x5a.xml 的几何/惯量参数为基准，手写实现 6 自由度手臂的标准
刚体动力学方程：M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ

全部逻辑封装在 X5aDynamics 一个类里，核心是四个子过程：
  - _rotation_matrix   : 旋转矩阵封装（固定姿态 × 关节旋转 = 子→父完整旋转）
  - _forward_recursion : 外推（正向递推，基→末，求各连杆 ω/ω̇/COM 加速度）
  - _backward_recursion: 内推（反向递推，末→基，牛顿/欧拉方程求关节力矩）
  - mass_matrix        : 质量矩阵封装（RNEA 列法）

逆动力学 τ = _rnea（外推+内推）结果 + armature·q̈ + damping·q̇。
""" 

import numpy as np  # 导入 NumPy，用于数组、矩阵和线性代数运算，支撑动力学计算
from inverse_solution import KinematicSolver  # 引入运动学求解器，提供 RPY/轴旋转矩阵和雅可比等几何工具

GRAVITY = 9.81   # 设置重力加速度为 9.81 m/s²，表示重力沿 -Z 方向作用于机械臂
DAMPING = 1.0    # 设置关节粘滞阻尼系数，模拟关节损耗和系统阻尼
ARMATURE = 0.01  # 设置关节 armature 惯量系数，模拟驱动器额外惯性项

# 每条 {pos, euler, axis, mass, com, fullinertia}，fullinertia 顺序 [ixx iyy izz ixy ixz iyz]。
_LINKS = [
    dict(pos=[0.0, 0.0, 0.0605], euler=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0],                 # 原点位置、固定姿态和旋转轴
         mass=0.066982, com=[0.0054231, -0.0080289, 0.017086],                                # 质量和质心位置，用于后续计算重力与惯性矩
         fullinertia=[0.00008, 0.00003, 0.00008, 0.0, 0.00001, 0.0]),                         # 惯量参数，表示绕质心的 3×3 惯量张量
    
    dict(pos=[0.02, 0.0, 0.04], euler=[0.0, 0.0, 0.0], axis=[0.0, 1.0, 0.0],                  # link2（joint2，绕 Y 轴）
         mass=1.0795, com=[-0.13237, 0.0020852, 0.00010549],
         fullinertia=[0.00051, 0.01599, 0.01605, 0.00001, -0.00004, 0.0]), 

    dict(pos=[-0.264, 0.0, 0.0], euler=[3.1416, 0.0, 0.0], axis=[0.0, 1.0, 0.0],              # link3（joint3，绕 Y 轴，euler 绕 X 转 180°)
         mass=0.54534, com=[0.18531, 0.00068376, -0.051638], 
         fullinertia=[0.00036, 0.00423, 0.00420, -0.00003, -0.00039, 0.00001]), 

    dict(pos=[0.245, 0.0, -0.056], euler=[0.0, 0.0, 0.0], axis=[0.0, 1.0, 0.0],               # link4（joint4，绕 Y 轴)
         mass=0.11714, com=[0.040231, 0.0044807, -0.035335],
         fullinertia=[0.00019, 0.00023, 0.00014, -0.00002, -0.00009, 0.00002]), 

    dict(pos=[0.06775, 0.0005, -0.0865], euler=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0],         # link5（joint5，绕 Z 轴）
         mass=0.63488, com=[0.003612, -1.5455e-05, 0.055214],
         fullinertia=[0.00083, 0.00082, 0.00026, 0.0, 0.00007, 0.0]),

    dict(pos=[0.02895, 0.0, 0.0865], euler=[-3.1416, 0.0, 0.0], axis=[1.0, 0.0, 0.0],         # link6（joint6，绕 X 轴，euler 绕 X 转 -180°） 
         mass=0.44089, com=[0.041697, 2.4368e-05, 0.00014464],
         fullinertia=[0.00038, 0.00028, 0.00050, 0.0, 0.0, 0.0]),

    dict(pos=[0.08657, 0.024896, -0.0002436], euler=[0.0, 0.0, 0.0], axis=[0.0, 1.0, 0.0],    # link7（夹爪手指 1，slide 沿 +Y）
         mass=0.064798, com=[-0.00035522, -0.007827, -0.0029883],
         fullinertia=[0.00002, 0.00003, 0.00003, 0.0, 0.0, 0.0]),

    dict(pos=[0.08657, -0.0249, -0.00024366], euler=[0.0, 0.0, 0.0], axis=[0.0, -1.0, 0.0],   # link8（夹爪手指 2，slide 沿 -Y）
         mass=0.064798, com=[-0.00035522, 0.0078277, 0.0024201],
         fullinertia=[0.00002, 0.00003, 0.00003, 0.0, 0.0, 0.0]),
]

class X5aDynamics:
    """ARX X5a 六自由度手臂的牛顿-欧拉动力学模型。手指（link7/link8）在闭合位姿
    折算进 link6 惯量，使 6×6 质量矩阵与 MuJoCo（手指冻结时）逐项一致。"""

    def __init__(self): 
        self.n = 6  # 设置机械臂自由度数为 6，代表 6 个主动关节
        arm = _LINKS[:6]  # 只保留前 6 个连杆参数，其余夹爪结构会在 link6 组合惯量中处理

        # 缓存各连杆常量：p=frame{i} 原点(父系)、R=固定旋转(子→父)、axis=关节轴、
        # com=质心(连杆系)、I=绕 COM 惯量(连杆系)、mass=质量
        self.p = np.array([l["pos"] for l in arm], dtype=float)                                      # 把每个连杆的原点位置转成数组
        self.R = np.array([KinematicSolver._matrix_from_rpy(l["euler"]) for l in arm], dtype=float)  # 把每个连杆固定姿态转为旋转矩阵
        self.axis = np.array([l["axis"] for l in arm], dtype=float)                                  # 提取每个关节的转动轴方向
        self.com = np.array([l["com"] for l in arm], dtype=float)                                    # 提取每个连杆的质心位置
        self.mass = np.array([l["mass"] for l in arm], dtype=float)                                  # 提取每个连杆质量
        self.I = np.array([self._fullinertia_to_matrix(l["fullinertia"]) for l in arm], dtype=float) # 把每个连杆惯量参数转成 3×3 惯量矩阵

        self.mass[5], self.com[5], self.I[5] = self._link6_inertia()                                 # 用 link6 与夹爪组合后的等效惯量覆盖末端连杆，保证末端总惯量一致
        self.damping = np.full(self.n, DAMPING)                                                      # 创建长度为 6 的阻尼向量，每个关节使用同一阻尼系数
        self.armature = np.full(self.n, ARMATURE)                                                    # 创建长度为 6 的 armature 向量，表示关节驱动器附加惯量

    @staticmethod
    def _fullinertia_to_matrix(fi):                      # 把 MuJoCo 的 6 元组 fullinertia 转换成 3×3 惯量矩阵
        ixx, iyy, izz, ixy, ixz, iyz = fi                # 解包惯量参数，分别对应 xx, yy, zz, xy, xz, yz 分量
        return np.array([[ixx, ixy, ixz],                # 构造惯量矩阵的第一行，包含 xx、xy、xz 项
                         [ixy, iyy, iyz],                # 构造惯量矩阵的第二行，包含 xy、yy、yz 项
                         [ixz, iyz, izz]], dtype=float)  # 构造惯量矩阵的第三行，包含 xz、yz、zz 项，并设为 float 类型

    @staticmethod
    def _composite(parts):                                                                        # 把多个子刚体的质量、COM、惯量合并成一个等效刚体，使用平行轴定理
        total = sum(m for m, _, _ in parts)                                                       # 汇总所有子刚体质量，得到组合刚体的总质量
        com = sum(m * np.asarray(c, float) for m, c, _ in parts) / total                          # 按质量加权计算组合刚体的 COM 位置
        i_origin = np.zeros((3, 3))                                                               # 初始化原点坐标系下的惯量矩阵，后续累加各部分贡献
        for m, c, ic in parts:                                                                    # 遍历每个子刚体，逐项叠加其惯量贡献
            c = np.asarray(c, float)                                                              # 将 COM 向量转成 NumPy 数组，便于矩阵运算
            i_origin += ic + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))                      # 使用平行轴定理累加相对原点的惯量贡献
        return total, com, i_origin - total * (np.dot(com, com) * np.eye(3) - np.outer(com, com)) # 返回组合后总质量、COM 和相对 COM 的惯量矩阵

    def _link6_inertia(self):                                                      # 计算 link6 与夹爪的组合惯量，并将其覆盖原 link6 的惯量参数
        parts = [(_LINKS[5]["mass"], _LINKS[5]["com"],                             # 先将原始 link6 的质量、COM 和惯量加入组合列表
                  self._fullinertia_to_matrix(_LINKS[5]["fullinertia"]))]          # 将 link6 的 fullinertia 转换为惯量矩阵并加入列表
        for f in _LINKS[6:]:                                                       # 遍历两个夹爪指的参数，逐个加入组合计算
            parts.append((f["mass"], np.asarray(f["pos"]) + np.asarray(f["com"]),  # 手指安装位置与其 COM 相加，得到相对 link6 的实际 COM
                          self._fullinertia_to_matrix(f["fullinertia"])))          # 将手指的惯量参数转成矩阵并加入组合列表
        return self._composite(parts)

    # 旋转矩阵
    def _rotation_matrix(self, q):                                                                 # 计算每个关节的完整旋转矩阵，完整旋转由固有姿态旋转矩阵与关节旋转矩阵乘积得到
        return [self.R[i] @ KinematicSolver._rot_axis(self.axis[i], q[i]) for i in range(self.n)]  # 逐关节构造坐标系变换矩阵，返回所有关节的子→父旋转矩阵

    # 外推（正向递推，基 → 末）
    def _forward_recursion(self, q, qd, qdd):
        Rf = self._rotation_matrix(q)        # 计算所有关节的完整旋转矩阵，作为后续坐标变换基础
        W = np.zeros((self.n, 3))            # 角速度矩阵，形状为 (n,3)
        WD = np.zeros((self.n, 3))           # 角加速度矩阵
        VC = np.zeros((self.n, 3))           # 线加速度矩阵
        w = np.zeros(3)                      # 初始化基座角速度为 0
        wd = np.zeros(3)                     # 初始化基座角加速度为 0
        v = np.array([0.0, 0.0, GRAVITY])    # 设置基座线加速度为 [0,0,g]，把重力在基座坐标中体现出来

        for i in range(self.n):                                                                    # 从第 1 个关节到第 n 个关节依次处理，执行递推计算
            a = self.axis[i]                                                                       # 当前关节的单位转动轴向量
            wp, wdp = Rf[i].T @ w, Rf[i].T @ wd                                                    # 将父坐标系中角速度和角加速度转换到当前关节本体坐标系
            w_i = wp + a * qd[i]                                                                   # 计算当前关节角速度，等于父系角速度加上本关节相对转动速度
            wd_i = wdp + np.cross(wp, a * qd[i]) + a * qdd[i]                                      # 计算当前关节角加速度，包含相对加速度和科氏项
            v_i = Rf[i].T @ (v + np.cross(wd, self.p[i]) + np.cross(w, np.cross(w, self.p[i])))    # 求当前关节的线速度，考虑平移和旋转耦合
            VC[i] = v_i + np.cross(wd_i, self.com[i]) + np.cross(w_i, np.cross(w_i, self.com[i]))  # 计算当前连杆 COM 的线加速度
            W[i], WD[i] = w_i, wd_i                                                                # 保存当前关节的角速度和角加速度到结果数组中
            w, wd, v = w_i, wd_i, v_i                                                              # 更新当前状态作为下一关节递推的父状态
        return W, WD, VC                                                                           # 返回角速度、角加速度和 COM 线加速度

    # 内推（反向递推，末 → 基）
    def _backward_recursion(self, q, W, WD, VC):
        Rf = self._rotation_matrix(q)                                         # 重建旋转矩阵，确保反向递推时坐标系一致
        tau = np.zeros(self.n)                                                # 初始化关节力矩数组，每个关节初始力矩为 0
        f_next = np.zeros(3)                                                  # 初始化向前传递的力向量，表示子链段传回来的作用力
        n_next = np.zeros(3)                                                  # 初始化向前传递的力矩向量，表示子链段传回来的作用力矩
        for i in range(self.n - 1, -1, -1):                                   # 从末端关节倒过来遍历，执行递推求关节扭矩
            F = self.mass[i] * VC[i]                                          # 计算当前连杆的线性惯性力，服从牛顿第二定律 F = m·a
            N = self.I[i] @ WD[i] + np.cross(W[i], self.I[i] @ W[i])          # 计算当前连杆的欧拉惯性力矩，含角加速度和离心项
            if i < self.n - 1:                                                # 如果当前并非最后一个关节，则需要接收下一关节传回的力和力矩
                f_n = Rf[i + 1] @ f_next                                      # 将下一关节传回的力转换到当前关节坐标系
                n_n = Rf[i + 1] @ n_next                                      # 将下一关节传回的力矩转换到当前关节坐标系
                p_next = self.p[i + 1]                                        # 读取下一连杆的原点偏移，用于形成力臂
            else:                                                             # 如果当前是末端关节，则没有下一个连杆，后继力和力矩都视为零
                f_n = n_n = p_next = np.zeros(3)                              # 末端处无后继力和力矩，设为零以终止递推
            f_i = F + f_n                                                     # 当前关节的总作用力 = 惯性力 + 上游传递力
            n_i = N + n_n + np.cross(self.com[i], F) + np.cross(p_next, f_n)  # 当前关节总力矩 = 欧拉惯性矩 + 传递力矩 + 质心力臂矩
            tau[i] = np.dot(n_i, self.axis[i])                                # 把当前总力矩投影到关节轴方向，得到该关节需要的驱动扭矩
            f_next, n_next = f_i, n_i                                         # 把当前计算结果更新，作为下一层递推的“传回数据”
            print(f"Joint {i+1}: Tau = {tau[i]}")  # 输出调试信息，显示每个关节的力、力矩和计算得到的扭矩
        return tau

    def _rnea(self, q, qd, qdd):                         # 纯刚体逆动力学入口函数，执行正向和反向递推得到 τ_rigid
        W, WD, VC = self._forward_recursion(q, qd, qdd)  # 正向递推求出各连杆的速度和加速度中间量
        return self._backward_recursion(q, W, WD, VC)    # 反向递推求出每个关节的刚体扭矩

    # 质量矩阵
    def mass_matrix(self, q):                            # 使用 RNEA 列法计算机械臂的质量矩阵 M(q)，质量矩阵由刚体惯量和 armature 惯量构成，计算方式为列法
        zero = np.zeros(self.n)                          # 创建长度为 n 的零向量，作为参考速度和加速度状态
        g0 = self._rnea(q, zero, zero)                   # 计算零速度/零加速度时的刚体扭矩，作为计算列法的基准
        M = np.zeros((self.n, self.n))                   # 初始化 n×n 的质量矩阵，等待逐列填充
        for j in range(self.n):                          # 逐个关节列进行计算，每一列代表在该关节施加单位加速度时的响应
            e = zero.copy()                              # 复制零向量作为当前单位基向量
            e[j] = 1.0                                   # 将第 j 个位置设为 1，表示只对第 j 个关节施加单位加速度
            M[:, j] = self._rnea(q, zero, e) - g0        # 计算第 j 列的惯性响应，并减去零状态基准得到质量矩阵列
        return M + np.diag(self.armature)                # 在刚体质量矩阵中加入关节 armature 对角项，得到最终质量矩阵

    # 公开接口（q/qd/qdd 均为长度 6 的 np 数组）
    def inverse_dynamics(self, q, qd, qdd):  # 对外提供逆动力学接口，返回含重力、科氏项和阻尼的驱动扭矩
        """逆动力学 τ = M·q̈ + C·q̇ + g + armature·q̈ + damping·q̇。"""  # 说明输出扭矩等于刚体逆动力学 + armature 惯性 + 阻尼项
        return self._rnea(q, qd, qdd) + self.armature * qdd + self.damping * qd  # 把刚体扭矩、关节惯性和阻尼项加起来得到总驱动扭矩

    def gravity(self, q):  # 计算重力项 g(q)，用于单独查看重力对各关节的影响
        """重力项 g(q) = RNEA(q, 0, 0)。"""  # 说明在零速度和零加速度下的 RNEA 输出就是纯重力项
        return self._rnea(q, np.zeros(self.n), np.zeros(self.n))  # 使用零速度零加速度状态求得每个关节的纯重力项

    def gravity_compensation(self, q, qd=None):  # 计算重力补偿力矩，抵消重力并叠加阻尼，用于零力保持/防软瘫
        """重力补偿力矩 τ = g(q) + damping·q̇（抵消重力、抑制残余速度的保持力矩）。"""  # 说明该力矩使机械臂克服重力保持静止
        qd = np.zeros(self.n) if qd is None else qd  # 未提供速度时默认零速度，仅抵消重力
        return self.gravity(q) + self.damping * qd  # 返回重力项与阻尼项之和作为保持力矩

    def coriolis(self, q, qd):  # 计算科氏/离心力项 C(q, qdot)·qdot，不含阻尼与 armature
        """科氏/离心力矢量 C·q̇（纯刚体，不含阻尼）。"""  # 说明该函数返回速度相关的非线性力项，等价于惯性耦合部分
        return self._rnea(q, qd, np.zeros(self.n)) - self.gravity(q)  # 计算零加速度时的速度相关扭矩，再减去重力项得到科氏/离心项

    def bias_force(self, q, qd):  # 计算偏置力 h = C·qdot + g，等价于 MuJoCo 中的 qfrc_bias
        """偏置力 h = C·q̇ + g（等价 MuJoCo qfrc_bias，不含阻尼）。"""  # 说明偏置力表示重力和速度相关项，且不包含阻尼项
        return self._rnea(q, qd, np.zeros(self.n))  # 在非零速度但零加速度下求 RNEA，得到偏置力 h

    def forward_dynamics(self, q, qd, tau):  # 计算正动力学，给定关节扭矩和状态，求 qddot
        """正动力学 q̈ = M⁻¹·(τ − C·q̇ − g − damping·q̇)，Cholesky 解。"""  # 说明正动力学通过解线性系统 M·qdd = tau - bias - damping·qd
        M = self.mass_matrix(q)  # 计算当前状态下的质量矩阵 M
        rhs = tau - self.bias_force(q, qd) - self.damping * qd  # 构造右侧向量，表示外力减去偏置力和阻尼项
        L = np.linalg.cholesky(M)  # 对 M 做 Cholesky 分解，得到下三角矩阵 L，使 M = L·L^T
        return np.linalg.solve(L.T, np.linalg.solve(L, rhs))  # 先解 L·y = rhs，再解 L^T·qdd = y，得到关节加速度

    def jacobian(self, q):  # 计算机械臂雅可比矩阵，描述末端位姿对关节角的灵敏度
        """雅可比矩阵（数值差分，委托 inverse_solution.KinematicSolver）。"""  # 说明这里调用运动学求解器提供的数值雅可比计算
        return KinematicSolver()._jacobian(np.asarray(q, dtype=float))  # 将关节角转成 float 数组并求雅可比，返回末端位置/姿态对关节的灵敏度矩阵

    def impedance_control(self, q, qd, pose_des, F_ext=None, M_imp=None, D_imp=None, K_imp=None):  # 笛卡尔阻抗控制入口
        """笛卡尔阻抗控制：在逆动力学基础上，让末端对外力呈现期望的质量-弹簧-阻尼特性。

        期望末端阻抗（x 为末端位姿 [x,y,z,roll,pitch,yaw]，F_ext 为末端外力/力矩 6 维）：
            M_imp·(ẍ − ẍ_des) + D_imp·(ẋ − ẋ_des) + K_imp·(x − x_des) = F_ext

        无外力时末端跟踪 pose_des；有外力 F_ext 时按 K_imp/D_imp/M_imp 柔顺让位。
        实现：阻抗关系反解末端期望加速度 → 雅可比映射到关节加速度 → 逆动力学求力矩。
        """  # 说明该函数把外力柔顺与计算力矩控制结合，返回可直接下发的关节力矩
        solver = KinematicSolver()                                        # 实例化运动学求解器，提供正运动学和雅可比
        x = solver.forward_kinematics(q)                                  # 当前末端位姿
        J = solver._jacobian(np.asarray(q, dtype=float))                   # 解析雅可比 ∂pose/∂q（6×6）
        e = x - np.asarray(pose_des, dtype=float)                         # 位姿误差 e = x − x_des
        e[3:] = solver._wrap_rpy(e[3:])                                   # 姿态误差归一化到 [-π,π]
        xd = J @ qd                                                       # 末端位姿速度 ẋ = J·q̇

        F_ext = np.zeros(6) if F_ext is None else np.asarray(F_ext, dtype=float)  # 末端外力，默认零

        if M_imp is None:                                                 # 默认阻抗参数（对角阵）
            M_imp = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])               # 平移惯量 1 kg、转动惯量 0.1
        if D_imp is None:
            D_imp = np.diag([30.0, 30.0, 30.0, 5.0, 5.0, 5.0])            # 平移阻尼 30、转动阻尼 5
        if K_imp is None:
            K_imp = np.diag([200.0, 200.0, 200.0, 20.0, 20.0, 20.0])      # 平移刚度 200 N/m、转动刚度 20

        # 1) 阻抗关系反解末端期望加速度（点到位保持 ẍ_des = 0）
        xdd_imp = np.linalg.solve(M_imp, F_ext - D_imp @ xd - K_imp @ e)  # ë = M_imp⁻¹·(F_ext − D_imp·ė − K_imp·e)

        # 2) 末端加速度映射到关节（忽略低速保持时的 J̇·q̇），并补偿外力经关节惯量的传递，使闭环严格等于期望阻抗
        M = self.mass_matrix(q)                                           # 总质量矩阵（含 armature）
        qdd_cmd = np.linalg.solve(J, xdd_imp) - np.linalg.solve(M, J.T @ F_ext)  # q̈ = J⁻¹·ë − M⁻¹·Jᵀ·F_ext

        # 3) 逆动力学：M·q̈ + C·q̇ + g + damping·q̇，得到关节力矩
        return self.inverse_dynamics(q, qd, qdd_cmd)                      # 返回含重力/科氏/阻尼补偿的关节力矩


__all__ = ["X5aDynamics"]  # 定义模块导出列表，仅暴露 X5aDynamics 类，方便其他文件按需导入
