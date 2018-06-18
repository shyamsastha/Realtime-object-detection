import multiprocessing
import ctypes

class MPVariable():
    """
    SHARED VARIABLES IN MULTIPROSESSING
    """
    running = multiprocessing.Value(ctypes.c_bool,True)
    frame_counter = multiprocessing.Value(ctypes.c_int,0)
    fps = multiprocessing.Value(ctypes.c_float,0.0)
    fps_frames = multiprocessing.Value(ctypes.c_int,0)
    fps_seconds = multiprocessing.Value(ctypes.c_float,0.0) # FPS ave in 5sec (fps_interval)
    fps_snapshot = multiprocessing.Value(ctypes.c_float,0.0) # FPS ave in 0.2sec 
    gpu_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    cpu_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    viz_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    lost_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    total_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    first_complete_time = multiprocessing.Value(ctypes.c_float,0.0)
