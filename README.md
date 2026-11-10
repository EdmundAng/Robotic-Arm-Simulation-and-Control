## A personal learning project to build a simple robotic arm (2-DOF Planar 2R Linkage) in simulation.

The simulation was first done fully in Python with NumPy and Matplotlib. 
Forward and inverse kinematics were derived by hand. 
The arm, workspace, joint angles were visualised in an interactive plot.
PID-control was implemented, along with drift and inertia as disturbance simulation.
<img width="1912" height="984" alt="image" src="https://github.com/user-attachments/assets/fac9c3e7-de49-451c-9493-543d9301fa80" />

The simulation was then recreated in MuJoCo.
The arm was modelled to simulate a servomotor with dynamics, such as position control (PD controller), joint friction and joint inertia.
Analysed the interaction of the arm with physics such as gravity, inertia, collision, friction.
Added motion control to the arm, velocity, acceleration, jerk limiting.
Demonstrated singularity, by using Jacobians to compute the joint velocities and trying to drive tip radially outward at constant speed, approaching elbow fully extended configuration (singularity configuration).
<img width="1892" height="732" alt="image" src="https://github.com/user-attachments/assets/fa2e761c-2395-4200-a2e6-b6a2a90dc7d0" />
