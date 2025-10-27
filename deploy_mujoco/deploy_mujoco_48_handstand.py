import time
import mujoco
import numpy as np
import torch
import yaml
import cv2
import os

# ---------- headless 渲染 ----------
os.environ["MUJOCO_GL"] = "osmesa"   # 无显示器必备；有显示器可删

# ---------- 你的工具函数 ----------
def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    grav = np.zeros(3)
    grav[0] = 2 * (-qz * qx + qw * qy)
    grav[1] = -2 * (qz * qy + qw * qx)
    grav[2] = 1 - 2 * (qw * qw + qz * qz)
    return grav

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return np.clip((target_q - q) * kp + (target_dq - dq) * kd, -25, 25)

# ---------- 配置 ----------
LEGGED_GYM_ROOT_DIR = "/home/hzy/go2/My_unitree_go2_gym-main/"
config_file = "go2.yaml"
with open(f"{LEGGED_GYM_ROOT_DIR}/deploy_mujoco/configs/{config_file}", "r") as f:
    cfg = yaml.safe_load(f)

policy_path   = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
xml_path      = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
sim_dt        = cfg["simulation_dt"]
ctrl_dec      = cfg["control_decimation"]
kps           = np.array(cfg["kps"],     dtype=np.float32)
kds           = np.array(cfg["kds"],     dtype=np.float32)
default_q     = np.array(cfg["default_angles"], dtype=np.float32)
ang_vel_scale = cfg["ang_vel_scale"]
dof_pos_scale = cfg["dof_pos_scale"]
dof_vel_scale = cfg["dof_vel_scale"]
lin_vel_scale = cfg["lin_vel_scale"]
action_scale  = cfg["action_scale"]
cmd_scale     = np.array(cfg["cmd_scale"], dtype=np.float32)
num_obs       = cfg["num_obs"]
num_act       = cfg["num_actions"]
cmd           = np.array(cfg["cmd_init"], dtype=np.float32)

# ---------- 初始化 ----------
action        = np.zeros(num_act, dtype=np.float32)
target_q      = default_q.copy()
obs           = np.zeros(num_obs, dtype=np.float32)
counter       = 0

model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)
model.opt.timestep = sim_dt

policy = torch.jit.load(policy_path)

# ---------- 视频 ----------
WIDTH, HEIGHT = 480, 480
FPS = int(1.0 / sim_dt)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("mujoco_simulation.mp4", fourcc, FPS, (WIDTH, HEIGHT))
renderer = mujoco.Renderer(model, WIDTH, HEIGHT)

# ---------- 可选：实时 viewer ----------
USE_VIEWER = True
if USE_VIEWER:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)

# ---------- 主循环 ----------
start = time.time()
sim_steps = int(cfg["simulation_duration"] / sim_dt)

for _ in range(sim_steps):
    # ------------ 控制 ------------
    if (time.time() - start) < 3:
        tau = pd_control(default_q, data.qpos[7:], kps, np.zeros_like(kds), data.qvel[6:], kds)
    else:
        tau = pd_control(target_q, data.qpos[7:], kps, np.zeros_like(kds), data.qvel[6:], kds)
    data.ctrl[:] = tau

    mujoco.mj_step(model, data)
    counter += 1

    # ------------ 策略 ------------
    if counter % ctrl_dec == 0:
        qj  = (data.qpos[7:] - default_q) * dof_pos_scale
        dqj = data.qvel[6:] * dof_vel_scale
        grav = get_gravity_orientation(data.qpos[3:7])
        omega = data.qvel[3:6] * ang_vel_scale

        period = 2.
        phase  = (counter * sim_dt) % period / period
        sin_p  = np.sin(2 * np.pi * phase)
        cos_p  = np.cos(2 * np.pi * phase)

        obs[:2] = [sin_p, cos_p]
        obs[2]  = 0
        obs[3:6] = omega
        obs[6:9] = grav
        obs[9:12] = cmd * cmd_scale
        obs[12:24] = qj
        obs[24:36] = dqj
        obs[36:48] = action

        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
        target_q = action * action_scale + default_q

    # ------------ 录像 ------------
    renderer.update_scene(data,camera="rec_cam")
    frame = renderer.render()
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # ------------ 可选 viewer ------------
    if USE_VIEWER:
        viewer.sync()
        time.sleep(max(0, sim_dt - (time.time() - start)))

# ---------- 收尾 ----------
out.release()
print("✅ 视频已保存为 mujoco_simulation.mp4")