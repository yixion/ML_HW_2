from autocar.simulation.track import Track 
import tkinter as tk
import numpy as np

class AutocarApp:
    def __init__(root_window, canvas_width=800, canvas_hight=600, canvas_padding=40):
       self.canvas = tk.Canvas(root_window, width=canvas_width, hight=canvas_hight)
       self.canvas.geometry(f"{canvas_width}x{canvas_hight}")



