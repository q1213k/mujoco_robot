# 三段生成平滑轨迹，用减速距离 v²/(2a) 判断何时开始减速，到点后锁定停止。

class PositionTracker:  # 定义一维位置跟踪器类，用于生成平滑的软目标位置
    """一维梯形速度轨迹：每个控制周期调用 step() 得到软目标位置。
    参数：  # 说明构造函数所需参数与含义
        max_velocity : 匀速段最大速度
        acceleration : 加速斜率
        deceleration : 减速斜率（默认与加速相同）
    """

    def __init__(self, max_velocity, acceleration, deceleration=None):  # 初始化位置跟踪器，设置速度和加速度参数
        self.max_velocity = max_velocity                  # 匀速段最大速度
        self.acceleration = acceleration                  # 加速斜率
        self.deceleration = deceleration or acceleration  # 减速斜率（默认同加速）
        self.lock_velocity = acceleration / 1000.0        # 低于此速度直接锁停，避免微小误差导致抖动
        self._brake = 0.5 / self.deceleration             # 减速距离系数 = 1/(2a)，用于估算制动距离
        self.reset(0.0, 0.0)                              # 初始位置与速度都设为零，准备开始跟踪

    def reset(self, position=0.0, velocity=0.0):  # 重置跟踪器状态，适用于切换目标时
        """重设当前估计位置/速度（换新目标时调用）。"""  # 说明该方法用于重置当前位置和当前速度
        self.position = position                          # 设置当前估计位置
        self.velocity = velocity                          # 设置当前估计速度

    def step(self, goal, dt):  # 在一个控制周期中推进位置跟踪状态，并返回当前软目标位置
        """推进一个控制周期，返回当前软目标位置（内部状态同步更新）。"""
        v = self.velocity                                  # 读取当前速度，用于下一步计算
        delta = goal - self.position                      # 计算当前目标与当前位置的误差

        # 到点：误差小于单步最小位移 a·dt² 时锁定，避免浮点极限环
        if abs(delta) <= self.deceleration * dt * dt:     # 如果距离小于制动阈值，则启动接近目标处理
            if abs(v) <= self.lock_velocity:              # 速度也小 → 直接停到目标
                self.position = goal                     # 将位置直接更新到目标值
                self.velocity = 0.0                      # 速度置零，确保收敛稳定
                return self.position                     # 返回锁定后的目标位置
            # 速度仍大 → 减速到 0
            v = max(v - self.deceleration * dt, 0.0) if v > 0 else min(v + self.deceleration * dt, 0.0)  # 根据速度方向进行线性减速，避免越过目标
            self.velocity = v                             # 保存减速后的速度
            self.position += v * dt                       # 用更新后的速度积分计算新位置
            return self.position                          # 返回当前软目标位置

        if v == 0:                                        # 静止：朝目标方向加速
            v += self.acceleration * dt * (1 if delta > 0 else -1)  # 如果目标在正向，则正向加速；否则负向加速
        elif delta > 0 and v > 0:                         # 同向(正)：加速/匀速/减速
            need = v * v * self._brake                     # 当前速度减速到 0 所需距离
            if abs(delta) > need:                          # 距离足够 → 继续加速或匀速
                if v < self.max_velocity:                 # 速度尚未达最大匀速值,则继续加速
                    v = min(v + self.acceleration * dt, self.max_velocity)
                elif v > self.max_velocity:               # 若速度超出限值，则回落到匀速极限
                    v = max(v - self.deceleration * dt, self.max_velocity)
            else:                                          # 距离不足 → 开始减速
                v = max(v - self.deceleration * dt, 0.0)  # 逐步降低速度，避免过冲
        elif delta < 0 and v < 0:                         # 同向(负)：对称处理
            need = v * v * self._brake                     # 负向速度减速到 0 所需距离
            if abs(delta) > need:                          # 距离足够 → 继续负向加速或匀速
                if v > -self.max_velocity:                # 负速度尚未达到最大允许值
                    v = max(v - self.acceleration * dt, -self.max_velocity)  # 继续减小速度（更负），但不超过最低值
                elif v < -self.max_velocity:              # 速度过快时需要回落到限值
                    v = min(v + self.deceleration * dt, -self.max_velocity)
            else:                                          # 距离不足 → 提前减速
                v = min(v + self.deceleration * dt, 0.0)  # 负方向速度收敛到零，避免越过目标
        else:                                             # 反向：先减速到 0
            v = max(v - self.deceleration * dt, 0.0) if v > 0 else min(v + self.deceleration * dt, 0.0)  # 根据当前速度方向先减速到零，再改变方向

        self.velocity = v                                 # 保存更新后的速度值
        self.position += v * dt                           # 用新速度积分位置，形成平滑的软目标位移
        return self.position                              # 返回当前时刻的软目标位置


