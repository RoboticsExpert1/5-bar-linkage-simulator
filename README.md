# 🤖 5-Bar Linkage Kinematics Simulator

Developed using Python, Tkinter, and Matplotlib, this **5-Bar Linkage Kinematics Simulator** is a powerful engineering tool that provides real-time calculation and visualization of Forward Kinematics (FK) and Inverse Kinematics (IK) through an intuitive GUI. It also features point-teaching capabilities and an obstacle avoidance trajectory generation algorithm.

---

## 📺 Project Walkthrough

Watch the simulator in action and see the 'Vibe Coding' development process in the YouTube video below:

[![5-Bar Linkage Simulator Demo](https://img.youtube.com/vi/xzPFbcFxMGE/0.jpg)](https://www.youtube.com/watch?v=xzPFbcFxMGE)
> **Click the image above to watch the "Vibe Coding" process for this 5 bar linkage.**



## ✨ Key Features

* **Real-time Kinematics Visualization:**
  * Dynamic rendering of the Workspace Cloud as Link Length parameters change.
  * Instant application of Forward Kinematics (Angles) and Inverse Kinematics (Coordinates).
* **Interactive Control Modes:**
  * **Jog Controls:** Precise step-by-step movement using a directional layout.
  * **Mouse Joystick & Sliders:** Intuitive drag-and-drop end-effector control directly on the plot or via independent axis sliders.
* **Teaching & Playback:**
  * Save specific coordinate points (Point A, Point B) and automatically move to them.
  * Simulate smooth reciprocating motion between the saved points.
* **Obstacle Avoidance Trajectory Generation:**
  * Place virtual obstacles within the workspace by defining X/Y coordinates, radius, and safety clearance.
  * Automatically calculate and display a safe, collision-free trajectory when moving between Point A and Point B.

---


## 🚀 How to Run

1. **Ensure Python is installed.**
2. **Install the required visualization library:**
   ```bash
   pip install numpy matplotlib
3. **Run the script:**
   ```bash
   python 5-Bar Linkage.py

---
**Author:** **SUCHEOL LEE** (Lee Sucheol Robotics Lab.)  
**Expertise:** Robotic Mechanism Design, Kinematics, CAD 19+ Years
