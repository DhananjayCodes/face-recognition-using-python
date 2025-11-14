import time
import tkinter as tk
from tkinter import StringVar

def update_time():
    if running:
        global elapsed_time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        root.after(1000, update_time)

def start_stopwatch():
    global start_time, running
    if not running:
        start_time = time.time() - elapsed_time
        running = True
        update_time()

def pause_stopwatch():
    global running
    running = False

def reset_stopwatch():
    global start_time, elapsed_time, running
    running = False
    elapsed_time = 0
    time_var.set("00:00:00")

# Initialize GUI
root = tk.Tk()
root.title("Stopwatch")

time_var = StringVar()
time_var.set("00:00:00")

time_label = tk.Label(root, textvariable=time_var, font=("Arial", 30))
time_label.pack()

start_button = tk.Button(root, text="Start", command=start_stopwatch)
start_button.pack()

pause_button = tk.Button(root, text="Pause", command=pause_stopwatch)
pause_button.pack()

reset_button = tk.Button(root, text="Reset", command=reset_stopwatch)
reset_button.pack()

start_time = 0
elapsed_time = 0
running = False

root.mainloop()