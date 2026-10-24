import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

LINK_LENGTHS = [0.2, 0.15]               # meters, matches the XML
JOINT_LIMITS = [(0, 180), (-120, 120)]   # degrees, relative to previous joint

# Arm Movement Control
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_R = 82
KEY_ADD, KEY_SUBTRACT, KEY_MULTIPLY = 334, 333, 332
KEY_W, KEY_A, KEY_S, KEY_D = 87, 65, 83, 68
target = {"m1": 90.0, "m2": 0.0}   # degrees, intialised with home position
step_size = 5.0

# Pacing: N physics steps per rendered frame so sim time ~ wall time
STEPS_PER_FRAME = 8                      
FRAME_DT = STEPS_PER_FRAME * model.opt.timestep # Target Timing Per Frame = 8 * 2 ms = 16 ms (62.5FPS ~ 60 FPS)

# Tracked points: look everything up once, outside the loop
BODIES = [("Base", "base"), ("Joint 1", "link1"), ("Joint 2", "link2")] # (label, name in xml)
body_ids = [model.body(name).id for _, name in BODIES] # get body's ID
bob_id = model.geom("bob").id # get geom id
NAMES = [label for label, _ in BODIES] + ["Bob (End-effector)"]

# Terminal printing / settle detection
PRINT_EVERY = 6                         # frames between samples (~10 Hz)
POS_TOL = 1e-4                           # m: max movement between samples to count as "still"
SETTLE_SAMPLES = 5                       # consecutive still samples to declare settled (~0.5 s)

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

def key_callback(keycode):
    global step_size
    if keycode == KEY_R:
        target["m1"], target["m2"] = 90.0, 0.0
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

    # validate joints are within boundary, y>0
    if y1 > 0 and y2 > 0:
        data.actuator("motor1").ctrl = np.deg2rad(target["m1"])
        data.actuator("motor2").ctrl = np.deg2rad(target["m2"])

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
            # get obstacle coordinates(x,y) and plug them into inverse kinematics, and update target if valid
            obstacle1_id = model.body("obstacle1").id
            obstacle_pos_xy = (data.xpos[obstacle1_id][0], data.xpos[obstacle1_id][1])
            result = inverse_kinematics(obstacle_pos_xy)
            if result is not None: target["m1"], target["m2"] = result

            forward_kinematics([target["m1"], target["m2"]])

            for _ in range(STEPS_PER_FRAME): 
                mujoco.mj_step(model, data) # physics advances

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
                    # print(format_line(data.time, pts))
                    # print_sensor_data()

        viewer.sync()   # once per frame

        remaining = FRAME_DT - (time.perf_counter() - frame_start) # subtract actual loop duration
        if remaining > 0:
            time.sleep(remaining) # sleep the remainder to achieve the target timing (FRAME_DT)