# NavRL Ubuntu 交付清单

该目录可直接复制到 **Ubuntu 20.04 + ROS Noetic** 主机；解压后按 `UBUNTU_DEPLOYMENT_GUIDE_CN.md` 执行。交付目标是将代码改至可部署、可验证、可继续现场联调的状态，不包含现场接线、标定取数、飞控解锁或实际飞行执行。

| 需求 | 已交付的代码和材料 | 现场仍需完成的事项 |
|---|---|---|
| 1. 新 PPO 推理适配 ROS | 训练端同源 `ppo.py`、`PPO.act()`、`step3-80new.pt`、结构校验 | 使用目标机 GPU/CPU 完成加载验证 |
| 2. 观测接口 | `observation.py` 固定状态 8 维、LiDAR 36x4、动态目标 5x10 | 对接实际定位、raycast/LiDAR 话题或服务 |
| 3. 动态目标检测 | YOLO 节点、深度检测节点、卡尔曼跟踪、位置/速度/尺寸服务 | 填写实际相机话题，验证模型权重和帧率 |
| 4. 深度与坐标标定 | 可编辑内外参 YAML、深度单位、`check_calibration.py` | 填入现场内参、相机-机体外参和坐标系约定 |
| 5. PX4/MAVROS | 控制接口、OFFBOARD 前置 setpoint、只读 `px4_preflight_check.py` | 飞控连接、坐标系/速度方向、低风险台架测试 |
| 6. 异常与安全 | 数据超时保持、NaN 防护、软件急停、安全盾数值健壮性修正 | 根据机体尺寸、速度和环境调整安全阈值 |
| 7. 依赖与部署 | Conda/ROS 依赖脚本、静态校验、统一 launch | 在目标 Ubuntu 上安装 ROS/PX4 依赖并编译 |
| 8. 常规文档 | Ubuntu 安装、参数、启动顺序、故障排查 | 记录最终硬件版本和启动命令 |
| 9. 实物文档 | `PHYSICAL_DEPLOYMENT_GUIDE_CN.md` | 严格按地面检查和逐级放开流程执行 |

## 审稿意见第七条支持材料

`SIM_REAL_GAP_ANALYSIS_CN.md` 给出仿真与实际应用差距的论文分析框架；`SIM_TO_REAL_BENCHMARK_PROTOCOL_CN.md` 给出状态估计/传感器融合、坐标标定、三组实验和计算开销的报告协议；`sim_real_gap.launch`、可控传感器扰动、JSONL 日志、策略基准、回放和汇总脚本提供可复现的量化链路。没有真机日志时，材料只能支撑“差距分析和 sim-to-sim 鲁棒性验证”，不能写成“已完成真机性能验证”。

本机验证产物位于 `benchmark_results/`：包含 Noetic CPU 策略前向基准、完整 headless Gazebo 的 clean/stress 控制日志，以及 `ACCEPTANCE_REPORT_CN.md` 的九项可视化验收矩阵。其解释边界见该目录的 `README_CN.md`；实机部分仍须按 `PHYSICAL_DEPLOYMENT_GUIDE_CN.md` 逐级完成。
