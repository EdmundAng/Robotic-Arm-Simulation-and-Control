import matplotlib
import matplotlib.pyplot as plt
from screeninfo import get_monitors
import math
import matplotlib.animation as animation
import numpy as np
from matplotlib.widgets import Slider
import itertools

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

LINK_LENGTHS = [2,1] # define link lengths
JOINT_LIMITS = [(0,180), (-60,60)] # relative to horizontal axis (0 deg)
joints = [(0,0)] * (len(LINK_LENGTHS) + 1) # initialise joints based on number of link + origin coordinate

def getJoint(idxLink, angle):
    '''
    function to calculate and save joint coordinates
    idxLink= 1 (first link), 2 (second link),...(1-indexed)
    angle in degree
    '''
    d = LINK_LENGTHS[idxLink-1]
    angle_rad = math.radians(angle)
    x_prev, y_prev = joints[idxLink-1] # previous joint as reference
    x, y = d*math.cos(angle_rad) + x_prev, d*math.sin(angle_rad) + y_prev
    joints[idxLink] = (x, y)

def animate(i):
    old_config = joints.copy() # save old config as shallow copy
    
    getJoint(idxLink=1, angle=slider1.val)
    getJoint(idxLink=2, angle=slider2.val + slider1.val)

    # add a function here to validate if all joints are within work area boundary
    if any(y < 0 for _, y in joints):   # if any joint is out of bounds
        joints[:] = old_config          # restore old config

    x, y = zip(*joints)
    line.set_data(x, y)
    return line,

plt.style.use('dark_background')            # Apply the built-in dark theme globally
plt.rcParams['figure.figsize'] = [10, 6]    # Set default window size to 10 inches wide by 6 inches high globally
fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-r')
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(np.arange(-6, 7, 1))
ax.set_yticks(np.arange(-2, 5, 1))
ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
ax.set_position([0.05, 0.15, 0.9, 0.83])   # [left, bottom, width, height]
move_figure_to_monitor(fig, monitor_index=1, y_offset=-100) # Move figure to your secondary monitor (Index 1)

# Slider 1
ax_slider1 = fig.add_axes([0.2, 0.08, 0.6, 0.04]) # left edge, bottom edge, width, height (relative)
lo, hi = JOINT_LIMITS[0]
slider1 = Slider(ax_slider1, 'Joint 1 (deg)', lo, hi, valinit=90, valstep=1)

# Slider 2
ax_slider2 = fig.add_axes([0.2, 0.02, 0.6, 0.04])
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

plt.show()