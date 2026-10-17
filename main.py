import matplotlib
import matplotlib.pyplot as plt
from screeninfo import get_monitors
import math
import matplotlib.animation as animation
import numpy as np
from matplotlib.widgets import Slider
import itertools
import time
from collections import deque

LINK_LENGTHS = [2,1.5] # define link lengths
JOINT_LIMITS = [(0,180), (-120,120)] # angles relative to previous joint
MANUAL_MODE = False
target_angles = [90, 0] # intial target angles (home position)
cur_angles = [90, 0] # running angles (initialise same as target)
newPositionClicked = True
K_P = 48
K_I = 64 
K_D = 11
I_LIMIT = 50 # anti-windup clamp on the integral
prev_time = time.time()
integral_t1, integral_t2 = 0, 0
prev_diff_t1, prev_diff_t2 = None, None
DRIFT = 5  # deg/s, simulate gravity-like drift disturbance (higher value makes it overshoot past target)
vel_t1 = vel_t2 = 0.0
INERTIA = True    # toggle: False = original no-inertia plant
B = 1.0           # viscous friction (1/s); lower = more momentum, more ringing

T_WINDOW = 10  # seconds of history shown on the plot
t0 = time.time()
time_hist = deque(maxlen=2000)
cur_hist = [deque(maxlen=2000), deque(maxlen=2000)]  # [joint1, joint2]
tgt_hist = [deque(maxlen=2000), deque(maxlen=2000)]
last_valid_target = list(target_angles)

def move_figure_to_monitor(fig, monitor_index=1, x_offset=0, y_offset=0):
    """
    Moves a figure to a specific monitor index.
    monitor_index=0 is usually the primary display.
    monitor_index=1 is usually the secondary display.
    """
    monitors = get_monitors()
    
    # Fallback to primary if the requested index doesn't exist
    if monitor_index >= len(monitors):
        print(f"Monitor {monitor_index} not found. Defaulting to primary.")
        monitor_index = 0
        
    target_monitor = monitors[monitor_index]
    
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    dpi = fig.dpi
    window_width = int(fig_width_inch * dpi)
    window_height = int(fig_height_inch * dpi)
    final_x = target_monitor.x + (target_monitor.width // 2) - (window_width // 2) + x_offset
    final_y = target_monitor.y + (target_monitor.height // 2) - (window_height // 2) + y_offset

    # Apply backend specific window movements
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry(f"+{final_x}+{final_y}")
    elif backend in ['QtAgg', 'Qt5Agg', 'GTK3Agg']:
        fig.canvas.manager.window.move(final_x, final_y)
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((final_x, final_y))
    else:
        try:
            fig.canvas.manager.window.geometry(f"+{final_x}+{final_y}")
        except AttributeError:
            print(f"Positioning not supported natively on backend: {backend}")

def forward_kinematics(angles_deg):
    if not angles_deg:
        return None
    
    L1, L2 = LINK_LENGTHS
    T1, T2 = (math.radians(a) for a in angles_deg)
    x1, y1 = L1 * np.cos(T1), L1 * np.sin(T1)
    x2, y2 = x1 + L2 * np.cos(T1 + T2), y1 + L2 * np.sin(T1 + T2)

    # validate joints are within boundary, y>0
    if y1 > 0 and y2 > 0:
        return [(0,0), (x1,y1), (x2,y2)]

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
    T2 = math.pi - angle_A # Supplementary angle, T2 found
    T1 = np.acos(x2 / np.sqrt(line_a_squared)) - angle_C # just another PT formula

    # account for inverted elbow configuration (alternate joint configuration for same target position)
    T1_inv = np.acos(x2 / np.sqrt(line_a_squared)) + angle_C
    T2_inv = -T2

    # convert to deg first
    T1, T2 = math.degrees(T1), math.degrees(T2)
    T1_inv, T2_inv = math.degrees(T1_inv), math.degrees(T2_inv)

    # validate if target within boundary, angle values are not NaN, joint angles within limits
    is_not_NaN = T1 == T1 and T2 and T2 and T1_inv == T1_inv and T2_inv == T2_inv
    is_valid_angles = (lo_1 <= T1 <= hi_1) and (lo_2 <= T2 <= hi_2) or (lo_1 <= T1_inv <= hi_1) and (lo_2 <= T2_inv <= hi_2)
    if y2 > 0 and is_not_NaN and is_valid_angles:
        print(f"Solution 1... Joint angles(deg): T1: {T1:.3f}, T2: {T2:.3f}")
        print(f"Solution 2... Joint angles(deg): T1_inv: {T1_inv:.3f}, T2_inv: {T2_inv:.3f}")

        # add custom gate rule (y=0 symmetry line), to force symmetry, to toggle alternate joint config
        # left of y=0 line -> solution 1; right of y=0 line -> solution 2
        if x2 <= 0:
            print(f"Solution 1 selected... Joint angles(deg): T1: {T1:.3f}, T2: {T2:.3f}\n")
            return ([T1, T2])
        else: # (x2 > 0)
            print(f"Solution 2 selected... Joint angles(deg): T1: {T1_inv:.3f}, T2: {T2_inv:.3f}\n")
            return ([T1_inv, T2_inv])
    else:
        print("Invalid joint config!\n")
        return None

def on_click(event):
    global target_angles, newPositionClicked, integral_t1, integral_t2, prev_diff_t1, prev_diff_t2, vel_t1, vel_t2
    # Check for Left Double Click only in graph axes only
    if event.inaxes and event.dblclick and event.button == 1:
        if MANUAL_MODE:
            print("WARNING! Manual Mode Engaged. Use Sliders to Control Arm.")
        else:
            print(f"\nLeft click detected at: X={event.xdata:.3f}, Y={event.ydata:.3f}")
            target_angles = inverse_kinematics((event.xdata, event.ydata))
            integral_t1 = integral_t2 = 0
            prev_diff_t1 = prev_diff_t2 = None
            vel_t1 = vel_t2 = 0.0
            newPositionClicked = True

def animate(i):
    global target_angles, cur_angles, last_valid_target, newPositionClicked, prev_time, integral_t1, integral_t2, prev_diff_t1, prev_diff_t2, vel_t1, vel_t2
    if MANUAL_MODE: target_angles = [slider1.val, slider2.val] # manual control with sliders
        
    # dt measured every frame, regardless of whether there's a target
    current_time = time.time()
    dt = min(current_time - prev_time, 0.1)  # clamp so a stall can't cause a huge step
    prev_time = current_time

    # this will step the angles toward targets
    cur_t1, cur_t2 = cur_angles
    if target_angles:
        target_t1, target_t2 = target_angles
        diff_t1, diff_t2 = target_t1 - cur_t1, target_t2 - cur_t2 # error

        integral_t1 = integral_t1 + diff_t1 * dt
        integral_t2 = integral_t2 + diff_t2 * dt
        # integral_t1 = max(-I_LIMIT, min(I_LIMIT, integral_t1 + diff_t1 * dt)) # prevent integral windup
        # integral_t2 = max(-I_LIMIT, min(I_LIMIT, integral_t2 + diff_t2 * dt))

        # derivative of error; zero on the first frame of a new target (no kick) or if dt == 0
        if prev_diff_t1 is None or dt <= 0:
            deriv_t1 = deriv_t2 = 0
        else:
            deriv_t1 = (diff_t1 - prev_diff_t1) / dt
            deriv_t2 = (diff_t2 - prev_diff_t2) / dt
        prev_diff_t1, prev_diff_t2 = diff_t1, diff_t2

        if round(diff_t1, 3) == 0 and round(diff_t2, 3) == 0: # target reached (gap tolerance)
            if newPositionClicked:
                print(f"Target Reached!, Current Joint angles(deg): T1: {cur_t1:.3f}, T2: {cur_t2:.3f}")
                # integral_t1 = integral_t2 = 0
                newPositionClicked = False
        else:
            u_t1 = K_P * diff_t1 + K_I * integral_t1 + K_D * deriv_t1
            u_t2 = K_P * diff_t2 + K_I * integral_t2 + K_D * deriv_t2

            if INERTIA:
                # u is an acceleration command; velocity carries momentum
                vel_t1 += (u_t1 - B * vel_t1) * dt
                vel_t2 += (u_t2 - B * vel_t2) * dt
                cur_t1 += (vel_t1 - DRIFT) * dt
                cur_t2 += vel_t2 * dt
            else:
                # u is a velocity command; no momentum
                cur_t1 += (u_t1 - DRIFT) * dt
                cur_t2 += u_t2 * dt
    else:
        prev_diff_t1 = prev_diff_t2 = None
    cur_angles = [cur_t1, cur_t2] # update global values of cur angles    
    target_joints = forward_kinematics(cur_angles)

    # --- log history for the angle-tracking figure ---
    if target_angles:   # keep last good target if a click was invalid
        last_valid_target = list(target_angles)
    time_hist.append(time.time() - t0)
    for j in range(2):
        cur_hist[j].append(cur_angles[j])
        tgt_hist[j].append(last_valid_target[j])

    # target_joints = forward_kinematics(target_angles) # this will teleport to target

    if target_joints:
        x, y = zip(*target_joints)
        line.set_data(x, y)
    
    return line,

def plot_work_envelope(ax, step=1):
    """Plot reachable end-effector positions given JOINT_LIMITS and the y >= 0 boundary."""
    th1 = np.arange(JOINT_LIMITS[0][0], JOINT_LIMITS[0][1] + step, step) # instead of slider, it just generates all points along the joint limit range
    th2 = np.arange(JOINT_LIMITS[1][0], JOINT_LIMITS[1][1] + step, step)
    T1, T2 = np.meshgrid(np.radians(th1), np.radians(th2)) # builds every combination of th1 and th2, the specific combination is read at each position/index, e.g. combination 1 = (T1[0,0], T2[0,0]) = (angle1, angle2)

    L1, L2 = LINK_LENGTHS
    # forward kinematics (joint 2 is relative to link 1, same as your sliders)
    x1, y1 = L1 * np.cos(T1), L1 * np.sin(T1)
    x2, y2 = x1 + L2 * np.cos(T1 + T2), y1 + L2 * np.sin(T1 + T2)

    # same validity rule as animate(): every joint must have y >= 0
    valid = (y1 >= 0) & (y2 >= 0) # make a boolean mask

    # plot the end effector points
    # use boolean mask on end effector coordinates
    ax.scatter(x2[valid], y2[valid], s=2, color='cyan', alpha=0.25, zorder=1,
               label='Work envelope')

    # optional: trace the boundary curves (the edges of the joint-limit rectangle)
    def edge(t1, t2):
        t1, t2 = np.radians(t1), np.radians(t2)
        x = L1*np.cos(t1) + L2*np.cos(t1 + t2)
        y = L1*np.sin(t1) + L2*np.sin(t1 + t2)
        m = y >= 0
        ax.plot(x[m], y[m], color='yellow', lw=1, zorder=2)

    lo2, hi2 = JOINT_LIMITS[1]
    lo1, hi1 = JOINT_LIMITS[0]
    edge(th1, np.full_like(th1, lo2, dtype=float))   # joint 2 at min
    edge(th1, np.full_like(th1, hi2, dtype=float))   # joint 2 at max
    edge(np.full_like(th2, lo1, dtype=float), th2)   # joint 1 at min
    edge(np.full_like(th2, hi1, dtype=float), th2)   # joint 1 at max

plt.style.use('dark_background')            # Apply the built-in dark theme globally
plt.rcParams['figure.figsize'] = [11, 9]    # Set default window size to 10 inches wide by 6 inches high globally
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-r')
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(np.arange(-6, 7, 1))
ax.set_yticks(np.arange(-2, 5, 1))
ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
ax.set_position([0.05, 0.15, 0.9, 0.83])   # [left, bottom, width, height]

# Slider 1
ax_slider1 = fig.add_axes([0.15, 0.08, 0.8, 0.04]) # left edge, bottom edge, width, height (relative)
lo, hi = JOINT_LIMITS[0]
slider1 = Slider(ax_slider1, 'Joint 1 (deg)', lo, hi, valinit=90, valstep=1)

# Slider 2
ax_slider2 = fig.add_axes([0.15, 0.02, 0.8, 0.04])
lo, hi = JOINT_LIMITS[1]
slider2 = Slider(ax_slider2, 'Joint 2 (deg)', lo, hi, valinit=0, valstep=1)

ani = animation.FuncAnimation(
    fig=fig, 
    func=animate, 
    frames=itertools.count(), # keeps it running forever (func effectively becomes the main running loop)
    interval=20,
    blit=True,
    cache_frame_data=False
)

plot_work_envelope(ax)

fig.canvas.mpl_connect('button_press_event', on_click)

# ---------- Figure 2: joint angles over time ----------
fig2, (ax_j1, ax_j2) = plt.subplots(2, 1, sharex=True, figsize=(8, 9))
fig2.suptitle("Joint angles over time")

cur_line1, = ax_j1.plot([], [], color='red', label='Current')
tgt_line1, = ax_j1.plot([], [], color='cyan', ls='--', drawstyle='steps-post', label='Target')
cur_line2, = ax_j2.plot([], [], color='red', label='Current')
tgt_line2, = ax_j2.plot([], [], color='cyan', ls='--', drawstyle='steps-post', label='Target')

for a, (lo_, hi_), name in zip((ax_j1, ax_j2), JOINT_LIMITS, ("Joint 1 (deg)", "Joint 2 (deg)")):
    a.set_ylim(lo_ - 10, hi_ + 10)
    a.set_ylabel(name)
    a.grid(True, color='#333333', linestyle='--', linewidth=0.5)
    a.legend(loc='upper right')
    # show joint limits as faint reference lines
    a.axhline(lo_, color='gray', lw=0.5)
    a.axhline(hi_, color='gray', lw=0.5)
ax_j2.set_xlabel("Time (s)")

def update_plot(frame):
    if not time_hist:
        return ()
    t = list(time_hist)
    cur_line1.set_data(t, list(cur_hist[0]))
    tgt_line1.set_data(t, list(tgt_hist[0]))
    cur_line2.set_data(t, list(cur_hist[1]))
    tgt_line2.set_data(t, list(tgt_hist[1]))
    ax_j2.set_xlim(max(0, t[-1] - T_WINDOW), max(T_WINDOW, t[-1]))  # scrolling window
    return cur_line1, tgt_line1, cur_line2, tgt_line2

ani2 = animation.FuncAnimation(
    fig2, update_plot, interval=100, blit=False, cache_frame_data=False
)

# Move figure to your secondary monitor (Index 1)
move_figure_to_monitor(fig, monitor_index=1, x_offset=-410, y_offset=-80)
move_figure_to_monitor(fig2, monitor_index=1, x_offset=540, y_offset=-80)

plt.show()