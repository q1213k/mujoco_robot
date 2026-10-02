import numpy as np                         # 导入 NumPy，用于数组/矩阵等数值计算
import mujoco                              # 导入 MuJoCo 物理引擎 Python 绑定
import mujoco_viewer                       # 导入自定义交互式查看器（渲染 + 仿真步进循环）

import inverse_solution                    # 导入自写的正/逆运动学求解模块
import motion_planner                      # 导入梯形速度轨迹规划器（PositionTracker）
import rrt_planner                         # 导入 RRT* 关节空间无碰撞路径规划器


def link6_to_gripper_pose(model) -> np.ndarray:
    """读取模型中 link6 到夹爪(TCP)的固定安装变换，返回 [x, y, z, roll, pitch, yaw]。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")  # 查找夹爪相对于link6的作用点 id
    return np.array([*model.site_pos[site_id], 0.0, 0.0, 0.0])  # 返回 TCP 相对 link6 的安装位置（安装姿态角记为 0）


# 场景参数
Q_READY = np.array([0.0, 1.2, 1.2, -1.57, 0.0, 0.0])  # L 型初始姿态（6 个关节角）
GRIP_OPEN, GRIP_CLOSE = 0.044, 0.0                    # 夹爪开合量：张开 / 闭合

PRE_GRASP_POSE = np.array([0.20,  0.20, 0.15, 0.0, np.pi / 2, 0.0])  # 预抓位姿（方块上方悬停）
PLACE_POSE     = np.array([0.00, -0.20, 0.07, 0.0, np.pi / 2, 0.0])  # 放置位姿（方块最终位置）

GRASP_DZ = 0.04                                        # 抓取时 TCP 相对方块中心的下沉量
MAX_VELOCITY = 3.0                                     # 关节最大角速度 (rad/s)
ACCELERATION = 30.0                                    # 关节加速度 (rad/s²)

DEMO_EXTERNAL_FORCE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 恒定叠加末端外力（默认全零）

class AdmittanceController:
    def __init__(self, dt, M=None, D=None, K=None):   # 构造：dt 为仿真步长，M/D/K 为可选阻抗参数
        self.dt = float(dt)                           # 保存积分步长（秒）
        # 6 维阻抗参数，分别对应平移 [x,y,z] 与旋转 [roll,pitch,yaw]
        if M is None:   # 虚拟惯量：平移 kg，旋转 kg·m²
            M = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])        # 默认虚拟惯量矩阵（对角）
        if D is None:   # 虚拟阻尼：平移 N·s/m，旋转 N·m·s/rad（近临界阻尼，回弹平滑无振荡）
            D = np.diag([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])     # 默认虚拟阻尼矩阵（对角）
        if K is None:   # 虚拟刚度：平移 N/m，旋转 N·m/rad（调低更“软”，拖动让位更明显）
            K = np.diag([80.0, 80.0, 80.0, 8.0, 8.0, 8.0])     # 默认虚拟刚度矩阵（对角）
        self.M = np.asarray(M, dtype=float)          # 保存虚拟惯量矩阵
        self.D = np.asarray(D, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.dx = np.zeros(6)    # 位置修正量积分状态（初值 0）
        self.dxd = np.zeros(6)   # 速度修正量积分状态（初值 0）

    def update(self, F_ext):
        """输入末端外力 F_ext(6维 [fx,fy,fz,tx,ty,tz]，N / N·m)，输出位置修正量 Δx(6维)。"""
        F_ext = np.asarray(F_ext, dtype=float)       # 规范化外力为 float 数组
        # 导纳方程解加速度：Δẍ = M⁻¹·(F_ext − D·Δẋ − K·Δx)
        ddx = np.linalg.solve(self.M, F_ext - self.D @ self.dxd - self.K @ self.dx)  # 解出加速度修正量 Δẍ
        # 半隐式欧拉积分（先用新速度再更新位置，数值更稳）
        self.dxd += ddx * self.dt                    # 速度修正量积分：Δẋ += Δẍ·dt
        self.dx += self.dxd * self.dt                # 位置修正量积分：Δx += Δẋ·dt
        return self.dx.copy()                        # 返回位置修正量（返回副本，保护内部状态）

    def reset(self):
        """清零积分状态（切换目标任务时调用）。"""
        self.dx = np.zeros(6)                        # 位置修正量归零
        self.dxd = np.zeros(6)                       # 速度修正量归零


class PickAndPlace(mujoco_viewer.CustomViewer):
    """用逆解 + RRT* 规划机械臂搬运正方体。"""

    def runBefore(self):
        '''首先定义抓取点、放置点、预抓点、预放点等关键位姿，然后逆解关节角创建动作列表，最后用 RRT* 规划相邻位姿间的无碰撞关节空间路径列表。'''
        mujoco.mj_forward(self.model, self.data)  # 首次前向计算，得到初始位姿/速度等派生量

        self.tool = link6_to_gripper_pose(self.model)  # 读取 link6 到夹爪(TCP)的固定安装变换
        target = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")].copy()  # 读取方块当前中心位置

        # 抓取位姿：与预抓位姿同姿态，水平对齐方块中心并下沉到抓取高度
        grasp_pose = PRE_GRASP_POSE.copy()  # 复制预抓位姿作为抓取位姿模板
        grasp_pose[:3] = [target[0], target[1], target[2] + GRASP_DZ]  # 水平对齐方块中心，z 下沉 GRASP_DZ

        # 放置前位姿：与放置位姿同姿态、同水平位置，抬升到预抓位姿的接近高度
        pre_place_pose = PLACE_POSE.copy()  # 复制放置位姿作为放置前位姿模板
        pre_place_pose[2] = PRE_GRASP_POSE[2]  # 抬升到预抓位姿的接近高度

        # 逆解关键路径点
        q_pre_grasp = inverse_solution.inverse_kinematics(PRE_GRASP_POSE, Q_READY, self.tool)  # 逆解预抓位姿关节角
        q_grasp = inverse_solution.inverse_kinematics(grasp_pose, q_pre_grasp, self.tool)  # 逆解抓取位姿关节角
        q_pre_place = inverse_solution.inverse_kinematics(pre_place_pose, q_grasp, self.tool)  # 逆解放置前位姿关节角
        q_place = inverse_solution.inverse_kinematics(PLACE_POSE, q_pre_place, self.tool)  # 逆解放置位姿关节角

        # 任务阶段序列：(关节角, 夹爪开合量)；夹爪开合变化为立即段，其余为 RRT* 运动段
        q0, v0 = self.data.qpos.copy(), self.data.qvel.copy()   # 备份初始状态（规划会步进改写）
        stages = [
            (q0[:6],      GRIP_OPEN),   # 起点：当前关节角
            (Q_READY,     GRIP_OPEN),   # L 型初始姿态
            (q_pre_grasp, GRIP_OPEN),   # 预抓（方块上方）
            (q_grasp,     GRIP_OPEN),   # 下降到抓取点
            (q_grasp,     GRIP_CLOSE),  # 闭合夹爪
            (q_pre_grasp, GRIP_CLOSE),  # 抬起
            (q_pre_place, GRIP_CLOSE),  # 搬运到放置点上方
            (q_place,     GRIP_CLOSE),  # 下降到放置点
            (q_place,     GRIP_OPEN),   # 张开夹爪（释放）
            (q_pre_place, GRIP_OPEN),   # 退回上方
            (Q_READY,     GRIP_OPEN),   # 回到 L 型初始姿态
        ]

        # RRT* 规划相邻路径点间的无碰撞关节空间路径，展开成动作列表
        planner = rrt_planner.RRTStarPlanner(self.model, self.data)  # 创建 RRT* 规划器

        # 规划时关闭 target 方块碰撞：方块平放地面与 floor 持续接触(ncon≠0)，
        # 会被 RRT* 误判为碰撞；关闭后 ncon 只反映机械臂自身与地面的碰撞
        cube_gid = self.model.body_geomadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")]  # 方块碰撞几何 id
        cube_cont = (self.model.geom_contype[cube_gid], self.model.geom_conaffinity[cube_gid])  # 备份方块碰撞类型/亲和
        self.model.geom_contype[cube_gid] = 0  # 关闭方块碰撞类型
        self.model.geom_conaffinity[cube_gid] = 0  # 关闭方块碰撞亲和

        steps = 150  # 每个 RRT* 运动段的步进步数
        self.actions = []  # 初始化动作列表（每项：(目标关节角, 步数, 夹爪开合量)）
        for (q_a, g_a), (q_b, g_b) in zip(stages, stages[1:]):  # 把相邻两个阶段两两配对
            if g_a != g_b:                                      # 夹爪状态不同为开合阶段 → 把夹爪切到下一状态
                self.actions.append((q_b, 1, g_b))              # 不走轨迹，直接切换夹爪进入下一状态
            if not np.array_equal(q_a, q_b):                    # 运动段 → RRT* 规划
                self.data.qpos[:], self.data.qvel[:] = q0, v0   # 恢复初始环境，避免前一段规划影响下一段
                path = planner.plan(q_a, q_b, max_iter=2000)    # RRT* 规划关节空间无碰撞路径
                self.actions += [(np.asarray(q), steps, g_b) for q in path[1:]]  # 把RRT*规划出的路径点展开成动作列表（跳过起点）

        # 恢复方块碰撞，保证运行时方块仍能落到地面 / 被夹爪吸附
        self.model.geom_contype[cube_gid], self.model.geom_conaffinity[cube_gid] = cube_cont  # 恢复方块碰撞类型/亲和
        self.data.qpos[:], self.data.qvel[:] = q0, v0          # 恢复初始状态，供仿真起步

        self.idx = 0  # 当前动作索引初始化
        self.grasp_offset = None  # 抓取偏移（夹爪闭合时记录 TCP 到方块的相对偏移）
        self.dt = self.model.opt.timestep  # 读取仿真时间步长
        self.trackers = [motion_planner.PositionTracker(MAX_VELOCITY, ACCELERATION) for _ in range(6)]  # 6 个关节的梯形速度轨迹器
        for j in range(6):  # 逐关节初始化轨迹器
            self.trackers[j].reset(self.data.qpos[j], 0.0)  # 轨迹器从当前关节角、零速度起步

        # 导纳控制：外力 → 位置修正量（AdmittanceController），再用解析雅可比映射到关节角
        self.admittance = AdmittanceController(self.dt)  # 创建导纳控制器
        self.solver = inverse_solution.KinematicSolver()  # 创建运动学求解器（提供解析雅可比）
        self.ee_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("link6", "link7", "link8")]  # 末端刚体 id（腕部 + 两手指）
        self.pert = self.handle.perturb  # viewer 扰动对象（读取拖动状态与所选刚体）

        print("规划完成，动作段数:", len(self.actions))  # 打印调试信息，显示总动作段数

    def runFunc(self):
        '''仿真循环：逐步推进路径列表，执行梯形速度轨迹 + 导纳控制。'''
        # 任务结束（idx 越界）后进入保持态：仍持有末位姿 Q_READY 并持续导纳控制，机械臂可继续被拖动让位
        if self.idx >= len(self.actions):
            q_target = Q_READY    # 保持位姿 = 任务结束回到的 L 型初始姿态
            total = 0             # 保持态无“步数”概念，置 0 占位
            grip = GRIP_OPEN      # 夹爪保持张开
            hold = True           # 保持态：不再推进任务索引
        else:
            q_target, total, grip = self.actions[self.idx]  # 取出当前动作：目标关节角/步数/夹爪开合量
            hold = False          # 运动态：正常推进任务

        self._virtual_grasp(grip)  # 按夹爪开合状态做虚拟抓取/释放

        if not hold and total <= 1:          # 夹爪动作：立即到位（仅运动期间）
            self.data.ctrl[:6] = q_target    # 直接下发 6 关节位置目标
            self.data.ctrl[6:8] = grip       # 下发夹爪开合目标
            self.idx += 1                    # 推进到下一动作
            return

        # 导纳控制：读取末端外力 → 位置修正量 Δx → 关节角修正量 Δq，叠加到位置控制目标上
        # 外力来源：拖动末端时 viewer 施加在所选刚体上的 xfrc_applied（世界系 6 维）；
        # 松手后 pert.active=0，外力自动归零使机械臂回弹；DEMO_EXTERNAL_FORCE 为可选的恒定叠加力。
        F_ext = DEMO_EXTERNAL_FORCE.copy()                                # 默认取恒定叠加力
        if self.pert.active and self.pert.select in self.ee_body_ids:     # 正在拖动末端刚体
            F_ext += self.data.xfrc_applied[self.pert.select]             # 叠加拖动产生的末端外力
        dx = self.admittance.update(F_ext)                                # 位置修正量 Δx
        dq = np.linalg.solve(self.solver._jacobian(self.data.qpos[:6]), dx)  # 关节角修正量 Δq

        arrived = True  # 到达标志（本动作是否已完成）
        for j in range(6):                   # 逐关节推进梯形速度轨迹，叠加导纳修正
            self.data.ctrl[j] = self.trackers[j].step(q_target[j], self.dt) + dq[j]  # 下发 = 轨迹位置 + 导纳修正
            if self.trackers[j].velocity != 0.0 or self.trackers[j].position != q_target[j]:  # 未到位则标记未到达
                arrived = False
        self.data.ctrl[6:8] = grip  # 下发夹爪开合目标

        if arrived and not hold:  # 运动态且全部关节到位 → 推进到下一动作
            print(f"动作 {self.idx + 1}/{len(self.actions)} 完成，夹爪开合量: {grip:.3f}")  # 打印调试信息
            self.idx += 1  # 推进到下一动作


    def _virtual_grasp(self, grip):
        """虚拟抓取：夹爪闭合时把正方体吸附到 TCP 跟随，张开释放。"""
        if not hasattr(self, "cube_qpos_adr"):  # 首次调用时初始化相关 id/地址
            for name in ("link7_collision", "link8_collision"):   # 关闭手指碰撞，避免推挤方块
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)  # 手指碰撞几何 id
                self.model.geom_contype[gid] = 0  # 关闭该手指碰撞类型
                self.model.geom_conaffinity[gid] = 0  # 关闭该手指碰撞亲和
            self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")  # TCP 位点 id
            self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")  # 方块刚体 id
            cube_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_free")  # 方块自由关节 id
            self.cube_qpos_adr = self.model.jnt_qposadr[cube_jnt]  # 方块位置自由度在 qpos 中的起始地址
            self.cube_dof_adr = self.model.jnt_dofadr[cube_jnt]  # 方块自由度在 qvel 中的起始地址

        tcp = self.data.site_xpos[self.tcp_id]  # 读取当前 TCP 世界位置

        if grip == GRIP_CLOSE:  # 夹爪闭合 → 吸附方块
            if self.grasp_offset is None:  # 首次闭合时记录偏移
                self.grasp_offset = self.data.xpos[self.cube_body_id] - tcp  # 记录方块中心相对 TCP 的偏移
            self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = tcp + self.grasp_offset  # 方块位置跟随 TCP
            self.data.qvel[self.cube_dof_adr:self.cube_dof_adr + 6] = 0.0  # 方块速度清零（完全吸附）
        else:  # 夹爪张开 → 释放
            self.grasp_offset = None  # 清除抓取偏移


if __name__ == "__main__":
    print("开始仿真")  # 打印启动提示
    PickAndPlace("./arx_x5a/scene.xml").run_loop()  # 创建实例并进入交互式仿真主循环
