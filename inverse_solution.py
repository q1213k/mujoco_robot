import math  # 导入数学库，用于三角函数和π常量
from typing import Iterable, Optional  # 引入可迭代对象类型和可选类型
import numpy as np  # 导入 NumPy，用于矩阵和向量计算


class KinematicSolver:  # 定义机械臂运动学求解类
# 这是 X5 机械臂的纯 Python 运动学实现版本

    def __init__(self):  # 类初始化函数
        self.joint_limits = np.array([  # 6 个关节的角度限制范围，防止逆解发散到不可用角度
            [-3.14, 3.14],  # 关节 1 限制范围
            [0.000, 3.14],  # 关节 2 限制范围
            [0.000, 3.14],  # 关节 3 限制范围
            [-1.57, 1.57],  # 关节 4 限制范围
            [-1.57, 1.57],  # 关节 5 限制范围
            [-3.14, 3.14],  # 关节 6 限制范围
        ], dtype=float)  # 统一使用 float 类型

    @staticmethod  # 静态方法，类似于普通函数，不需要借助对象调用
    def _rot_x(angle: float) -> np.ndarray:  # 生成绕 X 轴旋转矩阵
        c, s = math.cos(angle), math.sin(angle)  # 计算 cos 和 sin
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)  # 返回 3x3 旋转矩阵

    @staticmethod  # 静态方法
    def _rot_y(angle: float) -> np.ndarray:  # 生成绕 Y 轴旋转矩阵
        c, s = math.cos(angle), math.sin(angle)  # 计算 cos 和 sin
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)  # 返回 3x3 旋转矩阵

    @staticmethod  # 静态方法
    def _rot_z(angle: float) -> np.ndarray:  # 生成绕 Z 轴旋转矩阵
        c, s = math.cos(angle), math.sin(angle)  # 计算 cos 和 sin
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)  # 返回 3x3 旋转矩阵

    @staticmethod # 生成任意轴旋转矩阵（通用旋转）
    def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray: 
        axis = axis / (np.linalg.norm(axis) + 1e-12)  # 归一化旋转轴，避免零向量问题
        x, y, z = axis  # 提取轴分量
        c = math.cos(angle)  # 旋转角的余弦值
        s = math.sin(angle)  # 旋转角的正弦值
        v = 1.0 - c  # 计算 Rodrigues 公式中的中间变量
        return np.array(  # 返回任意轴旋转矩阵
            [
                [c + x * x * v, x * y * v - z * s, x * z * v + y * s],  # 第一行
                [y * x * v + z * s, c + y * y * v, y * z * v - x * s],  # 第二行
                [z * x * v - y * s, z * y * v + x * s, c + z * z * v],  # 第三行
            ],
            dtype=float,  # 指定数据类型为 float
        )

    @staticmethod  # 创建平移矩阵，用于位移变换
    def _transform(x: float, y: float, z: float) -> np.ndarray:  
        return np.array(  # 返回 4x4 平移矩阵
            [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]],
            dtype=float,
        )
    
    @staticmethod  # 从旋转矩阵提取 roll-pitch-yaw
    def _rpy_from_matrix(R: np.ndarray) -> np.ndarray:  
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)  # 计算用于判断奇异值的变量
        if sy > 1e-6:  # 非奇异情况
            roll = math.atan2(R[2, 1], R[2, 2])  # 提取 roll
            pitch = math.atan2(-R[2, 0], sy)  # 提取 pitch
            yaw = math.atan2(R[1, 0], R[0, 0])  # 提取 yaw
        else:  # 奇异情况
            roll = math.atan2(-R[1, 2], R[1, 1])  # 退化时的 roll 计算
            pitch = math.atan2(-R[2, 0], sy)  # 退化时的 pitch
            yaw = 0.0  # 退化时 yaw 设为 0
        return np.array([roll, pitch, yaw], dtype=float)  # 返回 [roll, pitch, yaw]

    @staticmethod # 从 roll-pitch-yaw 生成旋转矩阵
    def _matrix_from_rpy(rpy: Iterable[float]) -> np.ndarray:  
        roll, pitch, yaw = map(float, rpy)  # 将输入转换成浮点数并拆分
        return KinematicSolver._rot_z(yaw) @ KinematicSolver._rot_y(pitch) @ KinematicSolver._rot_x(roll)  # 组合 Rz*Ry*Rx

    @staticmethod  # 将 4x4 变换矩阵转成位姿 [x,y,z,r,p,y]
    def _matrix_to_pose(T: np.ndarray) -> np.ndarray:  
        pos = T[:3, 3]  # 提取平移向量
        rpy = KinematicSolver._rpy_from_matrix(T[:3, :3])  # 提取旋转矩阵对应的 rpy
        return np.concatenate([pos, rpy])  # 拼接成 6 维位姿向量

    @staticmethod  # 把欧拉角归一化到 [-π, π] 区间
    def _wrap_rpy(rpy: np.ndarray) -> np.ndarray:  
        return (rpy + np.pi) % (2.0 * np.pi) - np.pi  # 对角度做周期化处理

    @staticmethod  # 静态方法，不依赖某个具体的机械臂实例，只是定义一组固定参数
    def _joint_chain():  # 定义机械臂的关节链参数，取自 arx_x5a/x5a.xml 的精确偏移
        return [  # 返回一组关节配置字典
            {"axis": np.array([0.0, 0.0, 1.0]), "offset": np.array([0.0, 0.0, 0.0605]),         "fixed_rpy": np.array([0.0, 0.0, 0.0])},  # 关节 1：Z 轴旋转，偏移 z=0.0605
            {"axis": np.array([0.0, 1.0, 0.0]), "offset": np.array([0.02, 0.0, 0.04]),          "fixed_rpy": np.array([0.0, 0.0, 0.0])},  # 关节 2：Y 轴旋转，偏移 [0.02,0,0.04]
            {"axis": np.array([0.0, 1.0, 0.0]), "offset": np.array([-0.264, 0.0, 0.0]),         "fixed_rpy": np.array([3.1416, 0.0, 0.0])},  # 关节 3：Y 轴旋转，偏移 x=-0.264，绕 X 轴转 180°
            {"axis": np.array([0.0, 1.0, 0.0]), "offset": np.array([0.245, 0.0, -0.056]),       "fixed_rpy": np.array([0.0, 0.0, 0.0])},  # 关节 4：Y 轴旋转，偏移 [0.245,0,-0.056]
            {"axis": np.array([0.0, 0.0, 1.0]), "offset": np.array([0.06775, 0.0005, -0.0865]), "fixed_rpy": np.array([0.0, 0.0, 0.0])},  # 关节 5：Z 轴旋转，偏移 [0.06775,0.0005,-0.0865]
            {"axis": np.array([1.0, 0.0, 0.0]), "offset": np.array([0.02895, 0.0, 0.0865]),     "fixed_rpy": np.array([-3.1416, 0.0, 0.0])},  # 关节 6：X 轴旋转，偏移 [0.02895,0,0.0865]
        ]

    def _forward_transform(self, joint_angles: np.ndarray) -> np.ndarray:  # 传入NumPy 数组并返回NumPy 数组
        T = np.eye(4, dtype=float)  # 创建4x4单位矩阵，作为初始变换矩阵
        for joint_cfg, q in zip(self._joint_chain(), joint_angles):  # 把关节角度和关节参数打包成一对一的然后遍历
            fixed_rpy = joint_cfg["fixed_rpy"]  # 读取关节的固定安装姿态
            axis = joint_cfg["axis"]  # 读取关节转动轴
            offset = joint_cfg["offset"]  # 读取关节之间的相对平移

            # 矩阵乘法（@）—— 与 MuJoCo 约定一致：先平移(父系)、再固定姿态、最后绕关节轴旋转
            T = T @ KinematicSolver._transform(*offset)  # 先对连接段做平移（父系）

            T = T @ np.block([  # 再施加固定安装姿态，np.block：把几个小矩阵拼成一个大矩阵。
                    [KinematicSolver._matrix_from_rpy(fixed_rpy), np.zeros((3, 1))],  # 旋转部分
                    [np.zeros((1, 3)), np.array([[1.0]])],  # 平移部分为零
                ])
            T = T @ np.block([  # 最后根据当前关节角施加关节旋转，np.zeros((1, 3))：创建一个 1 行 3 列的零矩阵
                    [KinematicSolver._rot_axis(axis, q), np.zeros((3, 1))],  # 关节旋转矩阵
                    [np.zeros((1, 3)), np.array([[1.0]])],  # 4x4 扩展
                ])
        return T  # 返回总位姿变换矩阵

    def forward_kinematics(self, joint_angles: Iterable[float]) -> np.ndarray:  # 正向运动学：输入关节角，输出末端位姿
        q = np.asarray(joint_angles, dtype=float).reshape(6)  # 将输入转换为 6 维数组
        T = self._forward_transform(q)  # 计算总变换矩阵
        return KinematicSolver._matrix_to_pose(T)  # 转成 [x,y,z,roll,pitch,yaw] 形式返回

    def _jacobian(self, joint_angles: np.ndarray, eps: float = 1e-6) -> np.ndarray:  # 数值计算雅可比矩阵
        J = np.zeros((6, 6), dtype=float)  # 初始化 6x6 雅可比矩阵
        for i in range(6):  # 对每个关节分别计算一列
            q_plus = joint_angles.copy()  # 复制当前关节角
            q_minus = joint_angles.copy()  # 复制当前关节角
            q_plus[i] += eps  # 给第 i 个关节加一个小扰动
            q_minus[i] -= eps  # 给第 i 个关节减一个小扰动
            p_plus = self.forward_kinematics(q_plus)  # 计算扰动后末端位姿
            p_minus = self.forward_kinematics(q_minus)  # 计算扰动前末端位姿
            J[:, i] = (p_plus - p_minus) / (2.0 * eps)  # 近似导数，构造雅可比列
        return J  # 返回数值雅可比矩阵

    def inverse_kinematics(  # 逆向运动学：输入目标位姿，输出关节角
        self,
        target_pose: Iterable[float],
        initial_guess: Optional[Iterable[float]] = None,#如果没有传入值就使用默认的None表示空
        link6_to_gripper_pose: Optional[Iterable[float]] = None,  # link6 到夹爪的固定安装变换矩阵
    ) -> np.ndarray:
        pose = np.asarray(target_pose, dtype=float).reshape(6)  # 将输入目标位姿转成 6 维数组
        if link6_to_gripper_pose is not None:  # 如果输入变换矩阵，则先转换成 link6 位姿
            pose = gripper_pose_to_link6_pose(pose, link6_to_gripper_pose)  # 得到供逆解使用的 link6 目标位姿
        q = (  # 初始关节角估计
            np.asarray(initial_guess, dtype=float).reshape(6)  # 如果传入了 initial_guess（初始猜测的关节角），则使用它
            if initial_guess is not None  # 如果不传，则默认从零位姿开始
            else np.zeros(6, dtype=float)
        )
        for _ in range(200):  # 迭代最多 200 次，避免死循环
            current = self.forward_kinematics(q)  # 计算当前关节对应的位姿
            err = pose - current  # 计算误差 = 目标位姿 - 当前位姿
            err[3:6] = self._wrap_rpy(err[3:6])  # 对姿态误差做角度归一化，避免大角度跳变
            if np.linalg.norm(err) < 1e-8:  # 如果误差足够小，则收敛
                break  # 退出循环

            J = self._jacobian(q)  # 计算当前雅可比矩阵
            lam = 1e-4  # 阻尼系数，防止矩阵病态导致发散
            delta = np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ err)  # 解增量：最小二乘 + damping
            q = q + 0.8 * delta  # 按步长更新关节角

            for i in range(6):  # 对每个关节做范围限制
                lo, hi = self.joint_limits[i]  # 读出关节角限制
                if q[i] < lo:  # 如果角度低于下限
                    q[i] = lo  # 夹到下限
                elif q[i] > hi:  # 如果角度高于上限
                    q[i] = hi  # 夹到上限

        print("最终关节角:", q)  # 打印最终求解出的关节角
        return q  # 返回最终解出的关节角


def forward_kinematics(joint_angles: Iterable[float]) -> np.ndarray:  # 顶层函数：正向运动学接口
    return KinematicSolver().forward_kinematics(joint_angles)  # 创建求解器实例并调用正向解


def inverse_kinematics(  # 顶层函数：逆向运动学接口
    target_pose: Iterable[float],  # 目标夹爪或 link6 位姿
    initial_guess: Optional[Iterable[float]] = None,  # 逆解初始关节角
    link6_to_gripper_pose: Optional[Iterable[float]] = None,  # link6 到夹爪的固定安装变换
) -> np.ndarray:  # 返回 6 个关节角
    return KinematicSolver().inverse_kinematics(  # 创建求解器实例并执行逆解
        target_pose, initial_guess, link6_to_gripper_pose  # 传入目标位姿、初值和工具变换
    )


def gripper_pose_to_link6_pose(
    gripper_pose: Iterable[float],  # 基座坐标系中的夹爪目标位姿 [x, y, z, roll, pitch, yaw]
    link6_to_gripper_pose: Iterable[float],  # link6 到夹爪的固定位姿变换 [x, y, z, roll, pitch, yaw]
) -> np.ndarray:  # 返回基座坐标系中的 link6 目标位姿
    gripper_pose_array = np.asarray(gripper_pose, dtype=float).reshape(6)  # 将夹爪位姿转换成长度为 6 的浮点数组
    gripper_transform = np.eye(4, dtype=float)  # 创建基座到夹爪的 4x4 齐次变换矩阵
    gripper_transform[:3, :3] = KinematicSolver._matrix_from_rpy(
        gripper_pose_array[3:]  # 将夹爪的 roll、pitch、yaw 转换成旋转矩阵
    )  # 保存夹爪相对于基座的旋转部分
    gripper_transform[:3, 3] = gripper_pose_array[:3]  # 保存夹爪相对于基座的位置部分

    fixed_pose = np.asarray(link6_to_gripper_pose, dtype=float).reshape(6)  # 将固定安装变换转换成长度为 6 的浮点数组
    fixed_transform = np.eye(4, dtype=float)  # 创建 link6 到夹爪的固定 4x4 齐次变换矩阵
    fixed_transform[:3, :3] = KinematicSolver._matrix_from_rpy(fixed_pose[3:])  # 保存固定变换的旋转部分
    fixed_transform[:3, 3] = fixed_pose[:3]  # 保存固定变换的平移部分

    link6_transform = gripper_transform @ np.linalg.inv(fixed_transform)  # 用夹爪目标变换乘以固定变换的逆得到 link6 目标变换
    return KinematicSolver._matrix_to_pose(link6_transform)  # 将 link6 变换矩阵转换回 [x, y, z, roll, pitch, yaw]


__all__ = [
    "KinematicSolver",
    "forward_kinematics",
    "inverse_kinematics",
    "gripper_pose_to_link6_pose",
]  # 定义模块导出符号，方便外部 import
