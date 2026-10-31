from autocar.simulation.track import Track
from autocar.simulation.replay import ReplayFrame
import tkinter as tk
import numpy as np

class AutocarApp:
    def __init__(self,root_window, track, canvas_width=800, canvas_height=600, canvas_padding=40):
        # create canvas to draw the pattern, such as circle and rectrangle
        self.canvas = tk.Canvas(root_window, width=canvas_width, height=canvas_height, bg='#464B71')
        self.canvas.pack(fill="both", expand=True)
        # save initial parameter
        self.canvas_width = canvas_width
        self.canvas_height=canvas_height
        self.root = root_window
        self.track = track
        self.padding = canvas_padding
        # parameter for calculating real world position to canvas's position
        self.min_x = min(track.boundary[:,0])
        self.max_x = max(track.boundary[:,0])
        self.min_y = min(track.boundary[:,1])
        self.max_y = max(track.boundary[:,1])
        self.goal_boundary = np.array([track.goal_top_left,track.goal_bottom_right])
        x_scale = (canvas_width-2*canvas_padding)/(self.max_x - self.min_x)
        y_scale = (canvas_height-2*canvas_padding)/(self.max_y - self.min_y)
        self.scale = min(x_scale,y_scale)
        # replay state
        self.after_id = None # store the return id of root.after
        # create a button to change replay mode
        self.control_frame = tk.Frame(root_window)
        self.control_frame.pack(side="bottom",fill="x",)
        self.button_group = tk.Frame(self.control_frame)
        self.button_group.pack()
        self.mode_4d_button = tk.Button(self.button_group,
                                      text = "4D model replay",
                                        command=lambda: self.select_mode("4D"),
                                      activeforeground='#990000')
        self.mode_4d_button.pack(side="left",padx=5,pady=5)
        self.mode_6d_button = tk.Button(self.button_group,
                                      text = "6D model replay",
                                        command=lambda: self.select_mode("6D"),
                                      activeforeground='#990000')
        self.mode_6d_button.pack(side="left", padx=5,pady=5)

    def w2c(self,x,y):
        canvas_x = self.padding + (x - self.min_x)*self.scale
        canvas_y = self.padding + (self.max_y - y)*self.scale
        return canvas_x, canvas_y


    def draw_track(self):
        track = self.track
        track_points = []
        for i in range(track.boundary.shape[0]):
            x, y = self.w2c(track.boundary[i][0],
                            track.boundary[i][1])
            track_points.extend([x,y])
        self.canvas.create_line(*track_points, width=5, fill='#7CD5C7', joinstyle=tk.ROUND,capstyle=tk.ROUND)
        # draw goal area
        goal_boundary = self.goal_boundary
        start_x, start_y = self.w2c(goal_boundary[0][0],
                                    goal_boundary[0][1])
        end_x, end_y = self.w2c(goal_boundary[1][0],
                                goal_boundary[1][1])
        self.canvas.create_rectangle(start_x,start_y,end_x,end_y, fill='#118AB2', outline='#7CD5C6',width=5)
        # draw car
        x,y = self.w2c(track.start_position[0], track.start_position[1])
        r = 3.0 * self.scale
        self.car_id = self.canvas.create_oval(x-r,y-r, x+r, y+r, fill='#FBEC5D')
        # draw sensor
        self.sensor_line_ids = [
                self.canvas.create_line(0,0,0,0, fill="red", width=1),
                self.canvas.create_line(0,0,0,0, fill="green",width=1),
                self.canvas.create_line(0,0,0,0, fill="blue", width=1),
                ]
        # draw live data
        text = "current mode:\n"+"front distance:\n"+"right_distance:\n"+"left_distance:\n"+"position:\n"+"heading:\n"
        self.info_text_id = self.canvas.create_text(self.canvas_width-self.padding,self.padding,text=text, anchor='ne', fill='#F2F2ED', font=('Arial', 12))
        # draw trajectory
        self.trajectory_id = self.canvas.create_line(0,0,0,0,fill='#CDB4FF',width=2, joinstyle=tk.ROUND,capstyle=tk.ROUND)
        self.trajectory_points = []
        self.canvas.tag_lower(self.trajectory_id,self.car_id)

    def set_replay_frames(self,frames_4d, frames_6d):
        self.replay_frames = {
                "4D":frames_4d,
                "6D":frames_6d,
        }
        self.current_replay_mode = "4D"
    def start_replay(self,mode, delay_ms=100):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.frames = self.replay_frames[mode]
        self.cur_index = 0
        self.delay_ms = delay_ms
        self.trajectory_points = []
        self.canvas.coords(self.trajectory_id, 0,0,0,0)
        self.show_next_frame()
    def show_next_frame(self):
        if self.cur_index >= len(self.frames):
            self.after_id = None
            return
        # get frame
        f = self.frames[self.cur_index]
        # draw car
        x, y = self.w2c(f.x,f.y)
        self.trajectory_points.extend([x,y])
        r = 3.0*self.scale
        self.canvas.coords(self.car_id, x-r, y-r,x+r, y+r)
        # draw sensor line
        distances = [
                f.front_distance,
                f.right_distance,
                f.left_distance,
                ]
        angle_offsets = [
                0.0,
                -45.0,
                45.0,
                ]
        # calculate endpoint of sensor line 
        end_points = []
        for i in range(3):
           end_x = f.x + distances[i] * np.cos(np.deg2rad(f.heading_degrees+angle_offsets[i]))
           end_y = f.y + distances[i] * np.sin(np.deg2rad(f.heading_degrees+angle_offsets[i]))
           end_x, end_y = self.w2c(end_x,end_y)
           end_points.append([end_x,end_y])
        # update sensor line
        for line_id, (end_x,end_y) in zip(self.sensor_line_ids, end_points):
            self.canvas.coords(line_id, x, y, end_x, end_y)

        # update text on the screate
        text = (
            f"current mode: {self.current_replay_mode}\n"
            f"front distance: {f.front_distance:.2f}\n"
            f"right distance: {f.right_distance:.2f}\n"
            f"left distance: {f.left_distance:.2f}\n"
            f"position: {f.x:.2f}, {f.y:.2f}\n"
            f"heading: {f.heading_degrees:.2f}\n"
        )
        self.canvas.itemconfig(self.info_text_id, text=text)
        # draw trajectory
        if len(self.trajectory_points) >=4:
            self.canvas.coords(self.trajectory_id, *self.trajectory_points)
        self.cur_index += 1
        self.after_id = self.root.after(
                self.delay_ms,
                self.show_next_frame,) # put the new frame to schedule and get the id of new frame in the schedule

    def select_mode(self, mode):
        self.current_replay_mode = mode
        self.start_replay(mode = mode) 
