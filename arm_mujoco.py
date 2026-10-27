import time
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

LINK_LENGTHS = [0.2, 0.15]               # meters, matches the XML
JOINT_LIMITS = [(0, 180), (-120, 120)]   # degrees, relative to previous joint

# Arm Movement Control
KEY_BACKSPACE = 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_1, KEY_2, KEY_3 = 49, 50, 51
KEY_ADD, KEY_SUBTRACT, KEY_MULTIPLY = 334, 333, 332
target = {"m1": 0.0, "m2": 0.0}   # degrees
step_size = 5.0

KEY_SLASH = 47
followCube = False

# Velocity, Acceleration, Jerk Limiting
MAX_JOINT_SPEED = 120.0   # deg/s, velocity cap
A_MAX = 180.0             # deg/s², acceleration cap (180-720)
J_MAX = 500.0   # deg/s³  (accel ramps 0 -> A_MAX in 0.18 s), (500-20000, lower more rounded corners)
TAU   = 0.15      # s, how quickly velocity chases its desired value (smaller, chases harder)
MODE_NAMES = ["rate-limited", "acceleration-limited", "jerk-limited"]
mode  = 2        # 0 = rate-limited, 1 = accel-limited, 2 = jerk-limited
KEY_COMMA = 44
cmd = {"m1": 0.0, "m2": 0.0}   # angle sent to the actuators
vel = {"m1": 0.0, "m2": 0.0}   # commanded joint velocity (deg/s)
acc = {"m1": 0.0, "m2": 0.0}    # commanded joint acceleration (deg/s²)

# Pacing: N physics steps per rendered frame so sim time ~ wall time
STEPS_PER_FRAME = 8                      
FRAME_DT = STEPS_PER_FRAME * model.opt.timestep # Target Timing Per Frame = 8 * 2 ms = 16 ms (62.5FPS ~ 60 FPS)

# Tracked points: look everything up once, outside the loop
BODIES = [("Base", "base"), ("Joint 1", "link1"), ("Joint 2", "link2"), ("Cube", "obstacle1")] # (label, name in xml)
body_ids = [model.body(name).id for _, name in BODIES] # get body's ID
bob_id = model.geom("bob").id # get geom id
NAMES = [label for label, _ in BODIES] + ["Bob (End-effector)"]

# Terminal printing / settle detection
PRINT_EVERY = 6                         # frames between samples (~10 Hz)
POS_TOL = 1e-4                           # m: max movement between samples to count as "still"
SETTLE_SAMPLES = 5                       # consecutive still samples to declare settled (~0.5 s)

# Live plots of tip linear velocity and acceleration
PLOT_WINDOW_S = 5.0
PLOT_EVERY = 8
MAX_SAMPLES = int(PLOT_WINDOW_S / FRAME_DT)
VEL_DECIMALS = 3      # round to 0.001 m/s
ACC_DECIMALS = 2      # round to 0.01 m/s²
VEL_MIN_SPAN = 0.05   # y-axis never tighter than ±0.05 m/s
ACC_MIN_SPAN = 0.5    # y-axis never tighter than ±0.5 m/s²
t_buf = deque(maxlen=MAX_SAMPLES)
vel_buf = deque(maxlen=MAX_SAMPLES)      # each entry: [vx, vy, vz]
acc_buf = deque(maxlen=MAX_SAMPLES)      # each entry: [ax, ay, az]
BG = "#1e1e2e"
FG = "#cccccc"
GRID = "#444444"
C_X, C_Y, C_Z = "#00e5ff", "#ff4081", "#b2ff59"   # cyan, pink, lime
plt.ion()
fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
fig.patch.set_facecolor(BG)
def style_axis(ax, title, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white")
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.grid(True, color=GRID)
style_axis(ax_v, "Tip Linear Velocity", "velocity (m/s)")
style_axis(ax_a, "Tip Linear Acceleration", "acceleration (m/s²)")
ax_a.set_xlabel("sim time (s)", color=FG)
vel_lines = {
    "vx":  ax_v.plot([], [], color=C_X, lw=1.8, label="vx")[0],
    "vy":  ax_v.plot([], [], color=C_Y, lw=1.8, label="vy")[0],
    "vz":  ax_v.plot([], [], color=C_Z, lw=1.8, label="vz")[0],
    "|v|": ax_v.plot([], [], color="white", ls="--", lw=1.2, label="|v|")[0],
}
acc_lines = {
    "ax":  ax_a.plot([], [], color=C_X, lw=1.8, label="ax")[0],
    "ay":  ax_a.plot([], [], color=C_Y, lw=1.8, label="ay")[0],
    "az":  ax_a.plot([], [], color=C_Z, lw=1.8, label="az")[0],
    "|a|": ax_a.plot([], [], color="white", ls="--", lw=1.2, label="|a|")[0],
}
for ax in (ax_v, ax_a):
    ax.legend(loc="upper right", facecolor=BG, edgecolor="#888888", labelcolor=FG)
fig.tight_layout()

def fit_ylim(ax, arrays, floor):
    """Fit y-axis to the data, but never tighter than ±floor."""
    lo = min(-floor, *(arr.min() for arr in arrays))
    hi = max(floor, *(arr.max() for arr in arrays))
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

def update_plot():
    if not plt.fignum_exists(fig.number):   # user closed the plot window
        return
    t = np.array(t_buf)
    v = np.array(vel_buf)                   # shape (N, 3)
    a = np.array(acc_buf)                   # shape (N, 3)
    v_mag = np.linalg.norm(v, axis=1)
    a_mag = np.linalg.norm(a, axis=1)

    for i, key in enumerate(("vx", "vy", "vz")):
        vel_lines[key].set_data(t, v[:, i])
    vel_lines["|v|"].set_data(t, v_mag)

    for i, key in enumerate(("ax", "ay", "az")):
        acc_lines[key].set_data(t, a[:, i])
    acc_lines["|a|"].set_data(t, a_mag)

    # sharex=True, so setting xlim on one axis moves both
    ax_v.set_xlim(max(0.0, t[-1] - PLOT_WINDOW_S), max(PLOT_WINDOW_S, t[-1]))

    # y: fit to the visible data every redraw, with a minimum span
    fit_ylim(ax_v, [v, v_mag], VEL_MIN_SPAN)
    fit_ylim(ax_a, [a, a_mag], ACC_MIN_SPAN)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def get_points():
    """Return (4, 3) array: base, link1, link2 body origins + bob geom center."""
    return np.vstack([data.xpos[body_ids], data.geom_xpos[bob_id]])

def format_line(t, pts):
    cols = "  ".join(
        f"{n}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})" for n, p in zip(NAMES, pts)
    )
    return f"Frame Pos (Ground Truth): t={t:6.3f}s  {cols}"

def print_sensor_data():
    # print(f"Frame Pos (sensor): \t\t\t\t Joint1: {data.sensor("link1_pos").data}  Joint2: {data.sensor("link2_pos").data}  Bob (End-effector): {data.sensor("ee_pos").data}")
    print(f"Joint Position: Joint1: {data.sensor("q1").data}  Joint2: {data.sensor("q2").data}")
    print(f"Joint Velocity: Joint1: {data.sensor("dq1").data}  Joint2: {data.sensor("dq2").data}")
    print(f"Actuator Force: Joint1: {data.sensor("tau1").data}  Joint2: {data.sensor("tau2").data}")
    print(f"Tip IMU: {data.sensor("tip_imu").data}")
    print(f"Tip Linear Acceleration: {data.sensor("tip_acc").data}")
    print(f"Tip Linear Velocity: {data.sensor("tip_vel").data}, Magnitude |v|={np.linalg.norm(data.sensor("tip_vel").data):.4f} m/s")

def key_callback(keycode):
    global step_size, followCube, mode
    if keycode == KEY_BACKSPACE: # soft reset
        step_size = 5.0
        followCube = False
        mode = 2
        target["m1"], target["m2"] = 0.0, 0.0
        for k in cmd:
            cmd[k], vel[k], acc[k] = 0.0, 0.0, 0.0
    elif keycode == KEY_1: # preset pos 1 / home
        target["m1"], target["m2"] = 90.0, 0.0
    elif keycode == KEY_2: # preset pos 2
        target["m1"], target["m2"] = 30.0, -30.0
    elif keycode == KEY_3: # preset pos 3
        target["m1"], target["m2"] = 30.0, 0.0
    elif keycode == KEY_LEFT:
        target["m1"] = min(target["m1"] + step_size, JOINT_LIMITS[0][1])
    elif keycode == KEY_RIGHT:
        target["m1"] = max(target["m1"] - step_size, JOINT_LIMITS[0][0])
    elif keycode == KEY_UP:
        target["m2"] = min(target["m2"] + step_size, JOINT_LIMITS[1][1])
    elif keycode == KEY_DOWN:
        target["m2"] = max(target["m2"] - step_size, JOINT_LIMITS[1][0])
    elif keycode == KEY_ADD:
        step_size = min(180.0, step_size+10)
        print(f"Step Size: {step_size}")
    elif keycode == KEY_SUBTRACT:
        step_size = max(1.0, step_size-1)
        print(f"Step Size: {step_size}")
    elif keycode == KEY_MULTIPLY:
        step_size = 5.0
        print(f"Step Size: {step_size}")
    elif keycode == KEY_SLASH:
        followCube = not followCube
        print("Following Cube" if followCube else "Not Following Cube")        
    elif keycode == KEY_COMMA:
        mode = (mode + 1) % 3
        for k in cmd:
            vel[k] = 0.0
            acc[k] = 0.0
        print("Mode:", MODE_NAMES[mode])

def step_rate_limited():
    """velocity is capped, but jumps from 0 to the cap instantly."""
    max_step = MAX_JOINT_SPEED * FRAME_DT
    for k in cmd:
        step = np.clip(target[k] - cmd[k], -max_step, max_step)
        cmd[k] += step
        vel[k] = step / FRAME_DT

def step_accel_limited():
    """velocity can only change by A_MAX*dt per frame."""
    dt = FRAME_DT
    for k in cmd:
        e = target[k] - cmd[k]
        # fastest speed we can have and still stop in time (0.8 = safety margin)
        v_des = np.sign(e) * min(MAX_JOINT_SPEED, np.sqrt(2 * A_MAX * 0.8 * abs(e)))
        # move velocity toward v_des, but only by a limited amount
        vel[k] += np.clip(v_des - vel[k], -A_MAX * dt, A_MAX * dt)
        cmd[k] += vel[k] * dt
        # snap when arrived so it doesn't jitter around the target
        if abs(target[k] - cmd[k]) < 0.05 and abs(vel[k]) < A_MAX * dt:
            cmd[k], vel[k] = target[k], 0.0

def step_jerk_limited():
    """Acceleration can only change by J_MAX*dt per frame -> rounded velocity corners."""
    dt = FRAME_DT
    for k in cmd:
        e = target[k] - cmd[k]

        # Fastest speed we can have and still stop in time. Jerk limiting makes
        # braking take longer, so the stopping distance has an extra term:
        #   d = v²/(2A) + v*A/(2J)   ->   solve for v
        d = 0.5 * abs(e)   # 0.8 = safety margin, 0.5 if overshoot
        v_stop = (-A_MAX**2 / J_MAX + np.sqrt(A_MAX**4 / J_MAX**2 + 8 * A_MAX * d)) / 2
        v_des = np.sign(e) * min(MAX_JOINT_SPEED, v_stop)

        # acceleration needed to reach v_des, capped at A_MAX
        a_des = np.clip((v_des - vel[k]) / TAU, -A_MAX, A_MAX)

        # the jerk limit: acceleration may only move toward a_des by J_MAX*dt
        acc[k] += np.clip(a_des - acc[k], -J_MAX * dt, J_MAX * dt)

        vel[k] = np.clip(vel[k] + acc[k] * dt, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
        cmd[k] += vel[k] * dt

        # snap when arrived
        if abs(target[k] - cmd[k]) < 0.05 and abs(vel[k]) < 1.0:
            cmd[k], vel[k], acc[k] = target[k], 0.0, 0.0

def step_cmd_toward_target():
    if mode == 0:
        step_rate_limited()
    elif mode == 1:
        step_accel_limited()
    else:
        step_jerk_limited()

def forward_kinematics(angles_deg):
    lo_1, hi_1 = JOINT_LIMITS[0][0], JOINT_LIMITS[0][1]
    lo_2, hi_2 = JOINT_LIMITS[1][0], JOINT_LIMITS[1][1]
    is_valid_angles = (lo_1 <= angles_deg[0] <= hi_1) and (lo_2 <= angles_deg[1] <= hi_2)

    if not angles_deg or not is_valid_angles:
        return None
    
    L1, L2 = LINK_LENGTHS
    T1, T2 = (np.deg2rad(a) for a in angles_deg)
    x1, y1 = L1 * np.cos(T1), L1 * np.sin(T1)
    x2, y2 = x1 + L2 * np.cos(T1 + T2), y1 + L2 * np.sin(T1 + T2)

    # validate joints are within boundary, y>=0
    if y1 >= 0 and y2 >= 0:
        # data.actuator("motor1").ctrl = np.deg2rad(target["m1"])
        # data.actuator("motor2").ctrl = np.deg2rad(target["m2"])
        data.actuator("motor1").ctrl = np.deg2rad(cmd["m1"])
        data.actuator("motor2").ctrl = np.deg2rad(cmd["m2"])

def inverse_kinematics(target_pos):
    L1, L2 = LINK_LENGTHS
    lo_1, hi_1 = JOINT_LIMITS[0][0], JOINT_LIMITS[0][1]
    lo_2, hi_2 = JOINT_LIMITS[1][0], JOINT_LIMITS[1][1]
    x2, y2 = target_pos

    # compute first configuration
    line_a_squared = x2*x2 + y2*y2 # Euclidean distance
    with np.errstate(invalid="ignore"): # ignore error warning for unreachable targets
        angle_A = np.acos((L1*L1 + L2*L2 - line_a_squared) / (2 * L1 * L2)) # Law of Cosine
        angle_C = np.acos((line_a_squared + L1*L1 - L2*L2) / (2 * np.sqrt(line_a_squared) * L1)) # Law of Cosine
    T2 = np.pi - angle_A # Supplementary angle, T2 found
    T1 = np.acos(x2 / np.sqrt(line_a_squared)) - angle_C # just another PT formula

    # account for inverted elbow configuration (alternate joint configuration for same target position)
    T1_inv = np.acos(x2 / np.sqrt(line_a_squared)) + angle_C
    T2_inv = -T2

    # convert to deg first
    T1, T2 = np.rad2deg(T1), np.rad2deg(T2)
    T1_inv, T2_inv = np.rad2deg(T1_inv), np.rad2deg(T2_inv)

    # validate if target within boundary, angle values are not NaN, joint angles within limits
    is_not_NaN = T1 == T1 and T2 and T2 and T1_inv == T1_inv and T2_inv == T2_inv
    is_valid_angles = (lo_1 <= T1 <= hi_1) and (lo_2 <= T2 <= hi_2) or (lo_1 <= T1_inv <= hi_1) and (lo_2 <= T2_inv <= hi_2)
    if y2 > 0 and is_not_NaN and is_valid_angles:
        # print(f"Solution 1... Joint angles(deg): T1: {T1:.3f}, T2: {T2:.3f}")
        # print(f"Solution 2... Joint angles(deg): T1_inv: {T1_inv:.3f}, T2_inv: {T2_inv:.3f}")

        # add custom gate rule (y=0 symmetry line), to force symmetry, to toggle alternate joint config
        # left of y=0 line -> solution 1; right of y=0 line -> solution 2
        if x2 <= 0:
            # print(f"Solution 1 selected... Joint angles(deg): T1: {T1:.3f}, T2: {T2:.3f}\n")
            return ([T1, T2])
        else: # (x2 > 0)
            # print(f"Solution 2 selected... Joint angles(deg): T1: {T1_inv:.3f}, T2: {T2_inv:.3f}\n")
            return ([T1_inv, T2_inv])
    else:
        # print("Invalid joint config!\n")
        return None

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Valid kinematics before the first step (otherwise xpos is all zeros)
    mujoco.mj_forward(model, data)
    prev = get_points().copy()
    print(format_line(data.time, prev))

    still_count = 0
    settled = False
    frame = 0

    while viewer.is_running():
        frame_start = time.perf_counter()

        with viewer.lock():
            if followCube:
                # get obstacle coordinates(x,y) and plug them into inverse kinematics, and update target if valid
                obstacle1_id = model.body("obstacle1").id
                obstacle_pos_xy = (data.xpos[obstacle1_id][0], data.xpos[obstacle1_id][1])
                result = inverse_kinematics(obstacle_pos_xy)
                if result is not None: target["m1"], target["m2"] = result

            step_cmd_toward_target()
            forward_kinematics([target["m1"], target["m2"]])

            for _ in range(STEPS_PER_FRAME): 
                mujoco.mj_step(model, data) # physics advances

            # record sample every frame
            t_buf.append(data.time)
            vel_buf.append(np.round(data.sensor("tip_vel").data, VEL_DECIMALS))
            tip_acc = data.sensor("tip_acc").data + model.opt.gravity   # discounts gravity offset
            acc_buf.append(np.round(tip_acc, ACC_DECIMALS))   

            frame += 1
            if frame % PRINT_EVERY == 0: # Print joints cartesian coordinates
                pts = get_points()
                # Largest distance any tracked point moved since the last sample
                moved = np.linalg.norm(pts - prev, axis=1).max()
                prev = pts.copy()

                if moved < POS_TOL:
                    still_count += 1
                    if not settled:
                        # print(f"in tolerance zone... {format_line(data.time, pts)}")
                        if still_count >= SETTLE_SAMPLES:
                            settled = True
                            print(f"settled... {format_line(data.time, pts)}\n")
                else:
                    still_count = 0
                    settled = False          # moving again: resume printing
                    print("-----arm is moving-----")
                    # print(format_line(data.time, pts)) # ground truth frame pos
                    print_sensor_data()

        if frame % PLOT_EVERY == 0 and len(t_buf) > 1:
            update_plot()

        viewer.sync()   # once per frame

        remaining = FRAME_DT - (time.perf_counter() - frame_start) # subtract actual loop duration
        if remaining > 0:
            time.sleep(remaining) # sleep the remainder to achieve the target timing (FRAME_DT)