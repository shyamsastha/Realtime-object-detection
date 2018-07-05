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
    cap_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    gpu_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    cpu_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    vis_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    lost_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    total_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    first_complete_time = multiprocessing.Value(ctypes.c_float,0.0)
    sleep_interval = multiprocessing.Value(ctypes.c_float,0.005)
    vis_frame_counter = multiprocessing.Value(ctypes.c_int,0)
    vis_fps = multiprocessing.Value(ctypes.c_float,0.0)
    vis_fps_frames = multiprocessing.Value(ctypes.c_int,0)
    vis_fps_seconds = multiprocessing.Value(ctypes.c_float,0.0) # FPS ave in 5sec (fps_interval)
    send_proc_time = multiprocessing.Value(ctypes.c_float,0.0)
    vis_drop_frames = multiprocessing.Value(ctypes.c_int,0)
    vis_skip_rate = multiprocessing.Value(ctypes.c_float,0.0)

    """
    MULTI-PROCESSING PIPE
    """
    vis_in_con, det_out_con = multiprocessing.Pipe(duplex=False)


    def __del__(self):
        det_out_con.close()
        vis_in_con.close()
