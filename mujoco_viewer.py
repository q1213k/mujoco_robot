"""
MuJoCo 仿真查看器封装
====================
封装 mujoco.viewer.launch_passive，提供统一的自定义查看器基类。

用法：
    class MyApp(CustomViewer):
        def runBefore(self): ...   # 主循环开始前初始化一次
        def runFunc(self): ...     # 每个仿真步调用一次
"""

import time
import mujoco
import mujoco.viewer


class CustomViewer:
    """自定义查看器基类：加载模型并启动被动查看器。"""

    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30, spec=None):
        if spec is not None:                                  # 传入 spec 时直接编译（可先注入障碍物）
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def run_loop(self):
        """主循环：runBefore → 循环(runFunc + 步进 + 同步)。"""
        self.runBefore()
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)   # 前向计算位姿
            self.runFunc()                             # 用户控制逻辑
            mujoco.mj_step(self.model, self.data)      # 物理步进
            self.sync()                                # 刷新画面
            time.sleep(self.model.opt.timestep)        # 与真实时间对齐

    # 子类覆写入口
    def runBefore(self):
        pass

    def runFunc(self):
        pass
