import matplotlib
import matplotlib.pyplot as plt
from screeninfo import get_monitors
import math
import matplotlib.animation as animation
import numpy as np
from matplotlib.widgets import Slider
import itertools

LINK_LENGTHS = [2,1.5] # define link lengths
JOINT_LIMITS = [(0,180), (-90,90)] # angles relative to previous joint

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
    L1, L2 = LINK_LENGTHS
    T1, T2 = (math.radians(a) for a in angles_deg)
    x1, y1 = L1 * np.cos(T1), L1 * np.sin(T1)
    x2, y2 = x1 + L2 * np.cos(T1 + T2), y1 + L2 * np.sin(T1 + T2)
    return [(0,0), (x1,y1), (x2,y2)]

clicked_pos = (0, sum(LINK_LENGTHS)) # initialize home position as target
def inverse_kinematics():
    L1, L2 = LINK_LENGTHS
    lo_1, hi_1 = JOINT_LIMITS[0][0], JOINT_LIMITS[0][1]
    lo_2, hi_2 = JOINT_LIMITS[1][0], JOINT_LIMITS[1][1]
    x2, y2 = clicked_pos

    # compute first configuration
    line_a_squared = x2*x2 + y2*y2 # Euclidean distance
    angle_A = np.acos((L1*L1 + L2*L2 - line_a_squared) / (2 * L1 * L2)) # Law of Cosine
    angle_C = np.acos((line_a_squared + L1*L1 - L2*L2) / (2 * np.sqrt(line_a_squared) * L1)) # Law of Cosine
    T2 = math.pi - angle_A # Supplementary angle, T2 found
    T1 = np.acos(x2 / np.sqrt(line_a_squared)) - angle_C # just another PT formula

    # account for inverted elbow configuration (alternate joint configuration for same target position)
    T1_inv = np.acos(x2 / np.sqrt(line_a_squared)) + angle_C
    T2_inv = -T2
    
    # validate joint angles here
    # if first config joint angles are valid, try first configuration
    if x2 <= 0 and (lo_1 <= math.degrees(T1) <= hi_1) and (lo_2 <= math.degrees(T2) <= hi_2): 
        x1, y1 = x2 - L2 * np.cos(T1 + T2), y2 - L2 * np.sin(T1 + T2) # first configuration
        if y1 < 0 or y2 < 0: # work boundary check
            # print("Valid joints config, but out of bounds!")
            print("Invalid joint config!")
            return [(0,-1)]
        else:           
            print(f"Joint angles(deg): T1: {math.degrees(T1):.3f}, T2: {math.degrees(T2):.3f}")
            return [(0,0), (x1,y1), (x2,y2)]
    
    # else try inverted elbow configuration
    elif x2 > 0 and (lo_1 <= math.degrees(T1_inv) <= hi_1) and (lo_2 <= math.degrees(T2_inv)): 
        x1, y1 = L1 * np.cos(T1_inv), L1 * np.sin(T1_inv) # inverted configuration
        if y1 < 0 or y2 < 0: # work boundary check
            # print("Valid joints config, but out of bounds!")
            print("Invalid joint config!")
            return [(0,-1)]
        else:
            print(f"Joint angles: T1: {math.degrees(T1_inv):.3f}, T2: {math.degrees(T2_inv):.3f}")
            return [(0,0), (x1,y1), (x2,y2)]
    
    # else render it invalid
    else:
        print("Invalid joint config!")
        return [(0,-1)]

def on_click(event):
    global clicked_pos
    # Check for Left Double Click only in graph axes only
    if event.inaxes and event.dblclick and event.button == 1:
        print(f"Left click detected at: X={event.xdata:.3f}, Y={event.ydata:.3f}")
        clicked_pos = (event.xdata, event.ydata)

def animate(i):
    # use either forward or inverse kinematics, not simultaneously
    # joints = forward_kinematics([slider1.val, slider2.val])
    joints = inverse_kinematics()
    
    # validate joints are within boundary (boundary check, redundancy)
    if all(y >= 0 for _, y in joints): # only commit if all joints valid
        print(f"End effector at: ({joints[-1][0]:.3f}, {joints[-1][1]:.3f})\n")
        x, y = zip(*joints)
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
plt.rcParams['figure.figsize'] = [15, 9]    # Set default window size to 10 inches wide by 6 inches high globally
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-r')
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(np.arange(-6, 7, 1))
ax.set_yticks(np.arange(-2, 5, 1))
ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
ax.set_position([0.05, 0.15, 0.9, 0.83])   # [left, bottom, width, height]
move_figure_to_monitor(fig, monitor_index=1, y_offset=-80) # Move figure to your secondary monitor (Index 1)

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
    frames=itertools.count(), # keeps it running forever
    interval=20,
    blit=True,
    cache_frame_data=False
)

plot_work_envelope(ax)

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()