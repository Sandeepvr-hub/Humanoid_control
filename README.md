import mujoco
import mujoco.viewer
import numpy as np

XML = "XP_robot_with_actuators.xml"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

KP = 60
KD = 15

Kp_balance = 150
Kd_balance = 35

IDX = {model.actuator(i).name: i for i in range(model.nu)}

def base_pose():
    target = np.zeros(model.nu)

    if "hip_pitch_r" in IDX: target[IDX["hip_pitch_r"]] = 0.08
    if "hip_pitch_l" in IDX: target[IDX["hip_pitch_l"]] = 0.08
    if "knee_r" in IDX: target[IDX["knee_r"]] = 0.22
    if "knee_l" in IDX: target[IDX["knee_l"]] = 0.22
    if "ankle_pitch_r" in IDX: target[IDX["ankle_pitch_r"]] = -0.12
    if "ankle_pitch_l" in IDX: target[IDX["ankle_pitch_l"]] = -0.12

    return target

def walk_pose(t):
    target = base_pose()

    freq = 0.25
    phase = 2*np.pi*freq*t
    s = np.sin(phase)

    right_stance = s < 0

    swing = max(0, s)
    swing_l = max(0, -s)

    sway = 0.08 * s
    if "hip_roll_r" in IDX: target[IDX["hip_roll_r"]] = -sway
    if "hip_roll_l" in IDX: target[IDX["hip_roll_l"]] = sway

    if right_stance:
        target[IDX["hip_pitch_r"]] = 0.08
        target[IDX["knee_r"]] = 0.24
        target[IDX["ankle_pitch_r"]] = -0.14
    else:
        target[IDX["hip_pitch_r"]] = 0.08 + 0.15*s
        target[IDX["knee_r"]] = 0.22 + 0.5*swing
        target[IDX["ankle_pitch_r"]] = -0.12 - 0.1*s

    if not right_stance:
        target[IDX["hip_pitch_l"]] = 0.08
        target[IDX["knee_l"]] = 0.24
        target[IDX["ankle_pitch_l"]] = -0.14
    else:
        target[IDX["hip_pitch_l"]] = 0.08 - 0.15*s
        target[IDX["knee_l"]] = 0.22 + 0.5*swing_l
        target[IDX["ankle_pitch_l"]] = -0.12 + 0.1*s

    return target, right_stance

def get_tilt():
    quat = data.qpos[3:7]
    return 2*(quat[0]*quat[2] - quat[3]*quat[1])

def reset():
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[2] = 1.0
    data.qpos[3:7] = [1,0,0,0]
    mujoco.mj_forward(model, data)

reset()

model.dof_damping[:] = 10.0

t = 0
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        target, right_stance = walk_pose(t)

        qpos = data.qpos[7:7+model.nu]
        qvel = data.qvel[6:6+model.nu]

        torque = KP*(target - qpos) - KD*qvel

        tilt = get_tilt()
        tilt_rate = data.qvel[4]

        balance = -Kp_balance*tilt - Kd_balance*tilt_rate

        if "hip_pitch_r" in IDX: torque[IDX["hip_pitch_r"]] += balance
        if "hip_pitch_l" in IDX: torque[IDX["hip_pitch_l"]] += balance

        if right_stance:
            torque[IDX["ankle_pitch_r"]] -= 0.05
        else:
            torque[IDX["ankle_pitch_l"]] -= 0.05

        torque = np.clip(torque, -60, 60)

        data.ctrl[:] = 0.25*torque + 0.75*data.ctrl[:]

        data.qpos[1] -= 0.0015

        mujoco.mj_step(model, data)
        viewer.sync()

        t += dt
