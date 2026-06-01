import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FiveBarLinkageSim:
    def __init__(self, root):
        self.root = root
        self.root.title("5-Bar Linkage Simulator")
        self.root.geometry("1100x950") # Increased height slightly to accommodate Obstacles

        # --- Default Mechanism Parameters ---
        self.L0 = 100.0  # Base link length
        self.L1 = 120.0  # Link 1 length (Left crank)
        self.L2 = 120.0  # Link 2 length (Right crank)
        self.L3 = 180.0  # Link 3 length (Left floating link)
        self.L4 = 180.0  # Link 4 length (Right floating link)

        # Joint coordinates
        self.A = np.array([-self.L0 / 2.0, 0.0])  # Left base joint
        self.B = np.array([self.L0 / 2.0, 0.0])   # Right base joint
        
        # State Variables
        self.theta1 = 60.0  # Degrees
        self.theta2 = 120.0 # Degrees
        self.ee_x = 0.0
        self.ee_y = 150.0
        
        self.joystick_active = False

        # --- Teaching Variables ---
        self.teach_A = None
        self.teach_B = None
        self.is_reciprocating = False
        self.motion_t = 0.0
        self.motion_dir = 1.0
        
        # --- Obstacle & Trajectory Variables ---
        self.obstacles = [] # Stores dictionaries of obstacle data
        self.ab_path = []   # Waypoints for obstacle avoidance
        self.ab_path_distances = [] 
        self.ab_total_distance = 0.0

        # Build GUI Layout (UI 인스턴스 변수들 선언)
        self.setup_gui()
        
        # 초기 데이터를 입력창에 동기화하고 첫 화면 렌더링
        self.update_numerical_fields()
        self.run_fk()

    def setup_gui(self):
        # Left Panel for inputs (Control Panel)
        self.left_panel = ttk.Frame(self.root, padding="10")
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Right Panel for Matplotlib visualization
        self.right_panel = ttk.Frame(self.root, padding="10")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- 1. Link Lengths ---
        ttk.Label(self.left_panel, text="[ Link Lengths ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        f_len = ttk.Frame(self.left_panel)
        f_len.pack(fill=tk.X, pady=5)
        
        ttk.Label(f_len, text="Base (L0):").grid(row=0, column=0, sticky=tk.W)
        self.ent_l0 = ttk.Entry(f_len, width=8)
        self.ent_l0.insert(0, str(self.L0))
        self.ent_l0.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(f_len, text="Crank (L1=L2):").grid(row=1, column=0, sticky=tk.W)
        self.ent_l1 = ttk.Entry(f_len, width=8)
        self.ent_l1.insert(0, str(self.L1))
        self.ent_l1.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(f_len, text="Float (L3=L4):").grid(row=2, column=0, sticky=tk.W)
        self.ent_l3 = ttk.Entry(f_len, width=8)
        self.ent_l3.insert(0, str(self.L3))
        self.ent_l3.grid(row=2, column=1, padx=5, pady=2)

        # --- 2. Forward Kinematics Inputs (Angles) ---
        ttk.Label(self.left_panel, text="[ Forward Kinematics (deg) ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(15,5))
        f_fk = ttk.Frame(self.left_panel)
        f_fk.pack(fill=tk.X, pady=5)
        
        ttk.Label(f_fk, text="Theta 1 (Left):").grid(row=0, column=0, sticky=tk.W)
        self.ent_t1 = ttk.Entry(f_fk, width=8)
        self.ent_t1.insert(0, str(self.theta1))
        self.ent_t1.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(f_fk, text="Theta 2 (Right):").grid(row=1, column=0, sticky=tk.W)
        self.ent_t2 = ttk.Entry(f_fk, width=8)
        self.ent_t2.insert(0, str(self.theta2))
        self.ent_t2.grid(row=1, column=1, padx=5, pady=2)
        
        self.btn_fk = ttk.Button(f_fk, text="Apply FK", command=self.on_apply_fk)
        self.btn_fk.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # --- 3. Inverse Kinematics Inputs (Coordinates) ---
        ttk.Label(self.left_panel, text="[ Inverse Kinematics ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(15,5))
        f_ik = ttk.Frame(self.left_panel)
        f_ik.pack(fill=tk.X, pady=5)
        
        ttk.Label(f_ik, text="EE X:").grid(row=0, column=0, sticky=tk.W)
        self.ent_x = ttk.Entry(f_ik, width=8)
        self.ent_x.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(f_ik, text="EE Y:").grid(row=1, column=0, sticky=tk.W)
        self.ent_y = ttk.Entry(f_ik, width=8)
        self.ent_y.grid(row=1, column=1, padx=5, pady=2)
        
        self.btn_ik = ttk.Button(f_ik, text="Apply IK", command=self.on_apply_ik)
        self.btn_ik.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # --- 3.5. Jog Controls (Keyboard Arrow-Key Layout) ---
        ttk.Label(self.left_panel, text="[ Jog Controls ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(15,5))
        f_jog = ttk.Frame(self.left_panel)
        f_jog.pack(fill=tk.X, pady=5)

        ttk.Label(f_jog, text="Step (mm):").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.spin_jog_step = ttk.Spinbox(f_jog, from_=0.1, to=50.0, increment=0.5, width=6)
        self.spin_jog_step.set(5.0)
        self.spin_jog_step.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)

        f_arrows = ttk.Frame(f_jog)
        f_arrows.grid(row=1, column=0, columnspan=3, pady=10)

        self.btn_jog_up = ttk.Button(f_arrows, text="▲", width=5, command=lambda: self.jog_move(0, 1))
        self.btn_jog_left = ttk.Button(f_arrows, text="◀", width=5, command=lambda: self.jog_move(-1, 0))
        self.btn_jog_right = ttk.Button(f_arrows, text="▶", width=5, command=lambda: self.jog_move(1, 0))
        self.btn_jog_down = ttk.Button(f_arrows, text="▼", width=5, command=lambda: self.jog_move(0, -1))

        self.btn_jog_up.grid(row=0, column=1, pady=2)
        self.btn_jog_left.grid(row=1, column=0, padx=2)
        self.btn_jog_right.grid(row=1, column=2, padx=2)
        self.btn_jog_down.grid(row=2, column=1, pady=2)

        # --- 4. Teaching Controls ---
        ttk.Label(self.left_panel, text="[ Teaching Controls ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(15,5))
        f_teach = ttk.Frame(self.left_panel)
        f_teach.pack(fill=tk.X, pady=5)

        self.btn_save_a = ttk.Button(f_teach, text="Save A Point", command=self.save_point_A)
        self.btn_save_a.grid(row=0, column=0, padx=2, pady=2)
        self.btn_move_a = ttk.Button(f_teach, text="Move to A Point", command=self.move_to_A)
        self.btn_move_a.grid(row=0, column=1, padx=2, pady=2)

        self.btn_save_b = ttk.Button(f_teach, text="Save B Point", command=self.save_point_B)
        self.btn_save_b.grid(row=1, column=0, padx=2, pady=2)
        self.btn_move_b = ttk.Button(f_teach, text="Move to B Point", command=self.move_to_B)
        self.btn_move_b.grid(row=1, column=1, padx=2, pady=2)

        self.btn_ab_motion = ttk.Button(f_teach, text="Start A-B Reciprocating Motion", command=self.toggle_ab_motion)
        self.btn_ab_motion.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # --- [NEW] 4.5 Obstacle Controls ---
        ttk.Label(self.left_panel, text="[ Obstacles & Avoidance ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(15,5))
        f_obs = ttk.Frame(self.left_panel)
        f_obs.pack(fill=tk.X, pady=5)

        ttk.Label(f_obs, text="X Pos:").grid(row=0, column=0, sticky=tk.W)
        self.ent_obs_x = ttk.Entry(f_obs, width=6)
        self.ent_obs_x.insert(0, "0.0")
        self.ent_obs_x.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(f_obs, text="Y Pos:").grid(row=0, column=2, sticky=tk.W)
        self.ent_obs_y = ttk.Entry(f_obs, width=6)
        self.ent_obs_y.insert(0, "150.0")
        self.ent_obs_y.grid(row=0, column=3, padx=2, pady=2)

        ttk.Label(f_obs, text="Radius:").grid(row=1, column=0, sticky=tk.W)
        self.ent_obs_r = ttk.Entry(f_obs, width=6)
        self.ent_obs_r.insert(0, "20.0")
        self.ent_obs_r.grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(f_obs, text="Clearance:").grid(row=1, column=2, sticky=tk.W)
        self.ent_obs_c = ttk.Entry(f_obs, width=6)
        self.ent_obs_c.insert(0, "10.0")
        self.ent_obs_c.grid(row=1, column=3, padx=2, pady=2)

        self.btn_add_obs = ttk.Button(f_obs, text="Apply / Add Obstacle", command=self.add_obstacle)
        self.btn_add_obs.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.btn_clear_obs = ttk.Button(f_obs, text="Clear All", command=self.clear_obstacles)
        self.btn_clear_obs.grid(row=2, column=2, columnspan=2, pady=5, sticky=tk.EW)

        # --- 5. Interactive Controls ---
        ttk.Label(self.left_panel, text="[ Interactive Controls ]", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(15,5))
        
        self.chk_joy_var = tk.BooleanVar(value=False)
        self.chk_joy = ttk.Checkbutton(self.left_panel, text="Enable Mouse Joystick Mode", variable=self.chk_joy_var, command=self.toggle_joystick)
        self.chk_joy.pack(anchor=tk.W, pady=5)

        self.f_sliders = ttk.LabelFrame(self.left_panel, text="Position Sliders (Independent)", padding="5")
        self.f_sliders.pack(fill=tk.X, pady=5)

        ttk.Label(self.f_sliders, text="X Position:").pack(anchor=tk.W)
        self.sld_x = ttk.Scale(self.f_sliders, from_=-200, to=200, orient=tk.HORIZONTAL)
        self.sld_x.set(self.ee_x)
        self.sld_x.pack(fill=tk.X, pady=2)

        ttk.Label(self.f_sliders, text="Y Position:").pack(anchor=tk.W)
        self.sld_y = ttk.Scale(self.f_sliders, from_=0, to=350, orient=tk.VERTICAL)
        self.sld_y.set(self.ee_y)
        self.sld_y.pack(fill=tk.Y, pady=2, anchor=tk.CENTER)

        # 축 및 캔버스
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.sld_x.configure(command=self.on_slider_move)
        self.sld_y.configure(command=self.on_slider_move)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)

    # --- Kinematics Math ---
    def update_geometry_params(self):
        try:
            self.L0 = float(self.ent_l0.get())
            self.L1 = float(self.ent_l1.get())
            self.L2 = self.L1 
            self.L3 = float(self.ent_l3.get())
            self.L4 = self.L3 
            
            self.A = np.array([-self.L0 / 2.0, 0.0])
            self.B = np.array([self.L0 / 2.0, 0.0])
            return True
        except ValueError:
            messagebox.showerror("Error", "Invalid link length values entered.")
            return False

    def forward_kinematics(self, t1_deg, t2_deg):
        r1 = np.radians(t1_deg)
        r2 = np.radians(t2_deg)
        
        C = self.A + np.array([self.L1 * np.cos(r1), self.L1 * np.sin(r1)])
        D = self.B + np.array([self.L2 * np.cos(r2), self.L2 * np.sin(r2)])
        
        dist_CD = np.linalg.norm(D - C)
        if dist_CD > (self.L3 + self.L4) or dist_CD < abs(self.L3 - self.L4) or dist_CD == 0:
            return None, None, None, False
        
        a = (self.L3**2 - self.L4**2 + dist_CD**2) / (2 * dist_CD)
        h = np.sqrt(max(0.0, self.L3**2 - a**2))
        
        midpoint = C + a * (D - C) / dist_CD
        perp = np.array([-(D[1] - C[1]), D[0] - C[0]]) / dist_CD
        
        EE = midpoint + h * perp if (midpoint + h * perp)[1] >= midpoint[1] else midpoint - h * perp
        return EE, C, D, True

    def inverse_kinematics(self, x, y):
        P = np.array([x, y])
        
        dist_AP = np.linalg.norm(P - self.A)
        if dist_AP > (self.L1 + self.L3) or dist_AP < abs(self.L1 - self.L3):
            return None, None, False
            
        cos_alpha1 = (self.L1**2 + dist_AP**2 - self.L3**2) / (2 * self.L1 * dist_AP)
        alpha1 = np.arccos(np.clip(cos_alpha1, -1.0, 1.0))
        gamma1 = np.arctan2(P[1] - self.A[1], P[0] - self.A[0])
        t1 = gamma1 + alpha1 

        dist_BP = np.linalg.norm(P - self.B)
        if dist_BP > (self.L2 + self.L4) or dist_BP < abs(self.L2 - self.L4):
            return None, None, False
            
        cos_alpha2 = (self.L2**2 + dist_BP**2 - self.L4**2) / (2 * self.L2 * dist_BP)
        alpha2 = np.arccos(np.clip(cos_alpha2, -1.0, 1.0))
        gamma2 = np.arctan2(P[1] - self.B[1], P[0] - self.B[0])
        t2 = gamma2 - alpha2 
        
        return np.degrees(t1), np.degrees(t2), True

    # --- UI Interactions & Handlers ---
    def on_apply_fk(self):
        if not self.update_geometry_params(): return
        try:
            t1 = float(self.ent_t1.get())
            t2 = float(self.ent_t2.get())
            self.theta1, self.theta2 = t1, t2
            self.run_fk()
        except ValueError:
            messagebox.showerror("Error", "Invalid angles entered.")

    def on_apply_ik(self):
        if not self.update_geometry_params(): return
        try:
            x = float(self.ent_x.get())
            y = float(self.ent_y.get())
            self.ee_x, self.ee_y = x, y
            self.run_ik()
        except ValueError:
            messagebox.showerror("Error", "Invalid coordinates entered.")

    # --- Jog Execution Logic ---
    def jog_move(self, dx, dy):
        if not self.update_geometry_params(): return
        try:
            step = float(self.spin_jog_step.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid jog step size.")
            return

        target_x = self.ee_x + (dx * step)
        target_y = self.ee_y + (dy * step)

        t1, t2, success = self.inverse_kinematics(target_x, target_y)
        if success:
            self.ee_x = target_x
            self.ee_y = target_y
            self.theta1, self.theta2 = t1, t2
            EE, C, D, fk_succ = self.forward_kinematics(t1, t2)
            if fk_succ and EE is not None:
                self.update_numerical_fields()
                self.plot_mechanism(C, D, EE)
        else:
            messagebox.showwarning("Kinematics Error", "Target jog movement is outside the workspace boundaries.")

    # --- Obstacle Avoidance Logic ---
    def add_obstacle(self):
        try:
            x = float(self.ent_obs_x.get())
            y = float(self.ent_obs_y.get())
            r = float(self.ent_obs_r.get())
            c = float(self.ent_obs_c.get())
            self.obstacles.append({'x': x, 'y': y, 'r': r, 'clearance': c})
            self.update_trajectory_cache()
            self.run_fk()
        except ValueError:
            messagebox.showerror("Error", "Invalid obstacle parameters entered.")

    def clear_obstacles(self):
        self.obstacles = []
        self.update_trajectory_cache()
        self.run_fk()

    def update_trajectory_cache(self):
        if self.teach_A is not None and self.teach_B is not None:
            self.plan_trajectory(self.teach_A, self.teach_B)
        else:
            self.ab_path = []
            self.ab_path_distances = []
            self.ab_total_distance = 0.0

    def plan_trajectory(self, start, end):
        p1 = np.array(start)
        p2 = np.array(end)
        waypoints = self.get_avoidance_waypoints(p1, p2, depth=0)

        # Calculate distances along the path for smooth animation
        dists = [0.0]
        total_d = 0.0
        for i in range(len(waypoints)-1):
            d = np.linalg.norm(waypoints[i+1] - waypoints[i])
            total_d += d
            dists.append(total_d)

        self.ab_path = waypoints
        self.ab_path_distances = dists
        self.ab_total_distance = total_d

    def get_avoidance_waypoints(self, p1, p2, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return [p1, p2]

        for obs in self.obstacles:
            O = np.array([obs['x'], obs['y']])
            R = obs['r'] + obs['clearance']
            
            d_vec = p2 - p1
            l2 = np.dot(d_vec, d_vec)
            if l2 == 0: continue
            
            # Project obstacle center onto the line segment
            t = max(0.0, min(1.0, np.dot(O - p1, d_vec) / l2))
            proj = p1 + t * d_vec
            dist = np.linalg.norm(proj - O)
            
            # Check for intersection with the clearance zone
            if dist < R:
                if dist < 1e-5: # Path goes exactly through the center
                    n = np.array([-d_vec[1], d_vec[0]])
                    n = n / np.linalg.norm(n)
                else:
                    n = (proj - O) / dist
                
                # Create a via point safely outside the clearance zone
                via = O + n * (R + 1.0)
                
                # Recursively check the new sub-segments
                left_path = self.get_avoidance_waypoints(p1, via, depth+1)
                right_path = self.get_avoidance_waypoints(via, p2, depth+1)
                return left_path[:-1] + right_path
                
        return [p1, p2]

    # --- Teaching Execution Logic ---
    def save_point_A(self):
        self.teach_A = (self.ee_x, self.ee_y)
        self.update_trajectory_cache()
        self.run_ik() 
        print(f"Point A saved at: {self.teach_A}")

    def save_point_B(self):
        self.teach_B = (self.ee_x, self.ee_y)
        self.update_trajectory_cache()
        self.run_ik() 
        print(f"Point B saved at: {self.teach_B}")

    def move_to_A(self):
        if self.teach_A is None:
            messagebox.showwarning("Warning", "Point A has not been saved yet.")
            return
        self.go_to_point(*self.teach_A)

    def move_to_B(self):
        if self.teach_B is None:
            messagebox.showwarning("Warning", "Point B has not been saved yet.")
            return
        self.go_to_point(*self.teach_B)

    def go_to_point(self, x, y):
        t1, t2, success = self.inverse_kinematics(x, y)
        if success:
            self.ee_x, self.ee_y = x, y
            self.theta1, self.theta2 = t1, t2
            self.run_fk()
        else:
            messagebox.showwarning("Kinematics Error", "Saved point is out of reach.")

    def toggle_ab_motion(self):
        if self.is_reciprocating:
            self.is_reciprocating = False
            self.btn_ab_motion.config(text="Start A-B Reciprocating Motion")
        else:
            if self.teach_A is None or self.teach_B is None:
                messagebox.showwarning("Warning", "Please save both Point A and Point B first.")
                return
            
            # Ensure path is updated before moving
            self.update_trajectory_cache()
            if self.ab_total_distance == 0.0:
                messagebox.showwarning("Warning", "Point A and Point B are the same.")
                return

            self.is_reciprocating = True
            self.btn_ab_motion.config(text="Stop A-B Reciprocating Motion")
            self.motion_t = 0.0
            self.motion_dir = 1.0
            self.animate_ab_motion()

    def animate_ab_motion(self):
        if not self.is_reciprocating:
            return

        # Advance along the total path distance
        self.motion_t += 0.03 * self.motion_dir  # Slight speed adjustment for multi-waypoint paths
        if self.motion_t >= 1.0:
            self.motion_t = 1.0
            self.motion_dir = -1.0 # Reverse direction
        elif self.motion_t <= 0.0:
            self.motion_t = 0.0
            self.motion_dir = 1.0 # Reverse direction

        # Determine target X, Y on the planned path
        target_distance = self.motion_t * self.ab_total_distance
        target_x, target_y = self.teach_A[0], self.teach_A[1]
        
        for i in range(len(self.ab_path) - 1):
            if self.ab_path_distances[i] <= target_distance <= self.ab_path_distances[i+1]:
                seg_len = self.ab_path_distances[i+1] - self.ab_path_distances[i]
                if seg_len == 0: continue
                local_t = (target_distance - self.ab_path_distances[i]) / seg_len
                p1, p2 = self.ab_path[i], self.ab_path[i+1]
                point = p1 + local_t * (p2 - p1)
                target_x, target_y = point[0], point[1]
                break

        t1, t2, success = self.inverse_kinematics(target_x, target_y)
        if success:
            self.ee_x, self.ee_y = target_x, target_y
            self.theta1, self.theta2 = t1, t2
            EE, C, D, fk_succ = self.forward_kinematics(t1, t2)
            if fk_succ and EE is not None:
                self.update_numerical_fields()
                self.plot_mechanism(C, D, EE)
        else:
            self.toggle_ab_motion()
            messagebox.showwarning("Warning", "Trajectory segment goes out of bounds. Motion stopped.")
            return

        # Loop animation after 50ms
        self.root.after(50, self.animate_ab_motion)

    # --- Core Routines ---
    def run_fk(self):
        EE, C, D, success = self.forward_kinematics(self.theta1, self.theta2)
        if success and EE is not None:
            self.ee_x, self.ee_y = EE[0], EE[1]
            self.update_numerical_fields()
            self.plot_mechanism(C, D, EE)
        else:
            messagebox.showwarning("Kinematics Error", "Unreachable geometry layout under current angles.")

    def run_ik(self):
        t1, t2, success = self.inverse_kinematics(self.ee_x, self.ee_y)
        if success:
            self.theta1, self.theta2 = t1, t2
            EE, C, D, fk_succ = self.forward_kinematics(t1, t2)
            if fk_succ and EE is not None:
                self.update_numerical_fields()
                self.plot_mechanism(C, D, EE)
                return
        messagebox.showwarning("Kinematics Error", "Target coordinate is out of the workspace boundaries.")

    def on_slider_move(self, event=None):
        if not hasattr(self, 'sld_x') or not hasattr(self, 'sld_y') or not hasattr(self, 'ax'):
            return
            
        self.update_geometry_params()
        
        target_x = self.sld_x.get()
        target_y = self.sld_y.get()
        
        t1, t2, success = self.inverse_kinematics(target_x, target_y)
        if success:
            self.ee_x = target_x
            self.ee_y = target_y
            self.theta1, self.theta2 = t1, t2
            EE, C, D, fk_succ = self.forward_kinematics(t1, t2)
            if fk_succ and EE is not None:
                self.update_numerical_fields()
                self.plot_mechanism(C, D, EE)

    def toggle_joystick(self):
        self.joystick_active = self.chk_joy_var.get()

    def on_click(self, event):
        if not self.joystick_active or event.xdata is None or event.ydata is None: return
        self.drag_ee(event.xdata, event.ydata)

    def on_drag(self, event):
        if not self.joystick_active or event.xdata is None or event.ydata is None: return
        if event.button == 1: 
            self.drag_ee(event.xdata, event.ydata)

    def drag_ee(self, target_x, target_y):
        t1, t2, success = self.inverse_kinematics(target_x, target_y)
        if success:
            self.ee_x, self.ee_y = target_x, target_y
            self.theta1, self.theta2 = t1, t2
            
            EE, C, D, _ = self.forward_kinematics(t1, t2)
            self.update_numerical_fields()
            self.plot_mechanism(C, D, EE)

    def update_numerical_fields(self):
        self.ent_t1.delete(0, tk.END)
        self.ent_t1.insert(0, f"{self.theta1:.2f}")
        self.ent_t2.delete(0, tk.END)
        self.ent_t2.insert(0, f"{self.theta2:.2f}")
        
        self.ent_x.delete(0, tk.END)
        self.ent_x.insert(0, f"{self.ee_x:.2f}")
        self.ent_y.delete(0, tk.END)
        self.ent_y.insert(0, f"{self.ee_y:.2f}")
        
        if hasattr(self, 'sld_x') and hasattr(self, 'sld_y'):
            self.sld_x.configure(command="")
            self.sld_y.configure(command="")
            self.sld_x.set(self.ee_x)
            self.sld_y.set(self.ee_y)
            self.sld_x.configure(command=self.on_slider_move)
            self.sld_y.configure(command=self.on_slider_move)

    def calculate_workspace_cloud(self):
        x_grid = np.linspace(-abs(self.L1+self.L3)*1.5, abs(self.L1+self.L3)*1.5, 60)
        y_grid = np.linspace(0, abs(self.L1+self.L3)*1.5, 60)
        valid_x = []
        valid_y = []
        
        for gx in x_grid:
            for gy in y_grid:
                _, _, success = self.inverse_kinematics(gx, gy)
                if success:
                    valid_x.append(gx)
                    valid_y.append(gy)
        return valid_x, valid_y

    def plot_mechanism(self, C, D, EE):
        self.ax.clear()
        
        # Draw Shaded Workspace
        vw_x, vw_y = self.calculate_workspace_cloud()
        if vw_x:
            self.ax.scatter(vw_x, vw_y, color='lavender', s=12, label='Workspace')

        # Draw Obstacles
        for obs in self.obstacles:
            # Physical Obstacle
            phys_circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='indianred', fill=True, alpha=0.6)
            self.ax.add_patch(phys_circle)
            # Clearance Boundary
            clear_circle = plt.Circle((obs['x'], obs['y']), obs['r'] + obs['clearance'], color='orange', fill=False, linestyle='--', lw=1.5)
            self.ax.add_patch(clear_circle)

        # Link Assembly Trace Points
        self.ax.plot([self.A[0], self.B[0]], [self.A[1], self.B[1]], 'k-', lw=4, label='Base (L0)')
        self.ax.plot([self.A[0], C[0]], [self.A[1], C[1]], 'r-', lw=3, label='Link 1 (Crank)')
        self.ax.plot([C[0], EE[0]], [C[1], EE[1]], 'g-', lw=3, label='Link 3 (Float)')
        self.ax.plot([self.B[0], D[0]], [self.B[1], D[1]], 'b-', lw=3, label='Link 2 (Crank)')
        self.ax.plot([D[0], EE[0]], [D[1], EE[1]], 'm-', lw=3, label='Link 4 (Float)')

        # Joints markers
        self.ax.plot(self.A[0], self.A[1], 'ko', ms=8)
        self.ax.plot(self.B[0], self.B[1], 'ko', ms=8)
        self.ax.plot(C[0], C[1], 'ro', ms=6)
        self.ax.plot(D[0], D[1], 'bo', ms=6)
        self.ax.plot(EE[0], EE[1], 'md', ms=10, label='End-Effector')

        # --- Draw Teaching Points and Avoidance Trajectory ---
        if self.teach_A is not None and self.teach_B is not None:
            if len(self.ab_path) > 1:
                path_x = [p[0] for p in self.ab_path]
                path_y = [p[1] for p in self.ab_path]
                self.ax.plot(path_x, path_y, color='orange', linestyle='--', lw=2, label='Safe Trajectory')

        if self.teach_A is not None:
            self.ax.plot(self.teach_A[0], self.teach_A[1], marker='*', color='gold', ms=15, markeredgecolor='black')
            self.ax.text(self.teach_A[0]+5, self.teach_A[1]+5, 'A', fontsize=12, weight='bold', color='black')

        if self.teach_B is not None:
            self.ax.plot(self.teach_B[0], self.teach_B[1], marker='*', color='cyan', ms=15, markeredgecolor='black')
            self.ax.text(self.teach_B[0]+5, self.teach_B[1]+5, 'B', fontsize=12, weight='bold', color='black')

        # Formatting configurations
        self.ax.set_title("5-Bar Kinematics Viewspace")
        self.ax.set_xlabel("X Axis (mm)")
        self.ax.set_ylabel("Y Axis (mm)")
        self.ax.grid(True, linestyle=':')
        
        self.ax.set_aspect('equal', adjustable='box')
        
        limit_val = (self.L1 + self.L3) * 1.3
        self.ax.set_xlim(-limit_val, limit_val)
        self.ax.set_ylim(-50, limit_val)
        
        # Avoid duplicate legend entries using a dict mapping
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = FiveBarLinkageSim(root)
    root.mainloop()