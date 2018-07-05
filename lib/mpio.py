import threading
import time
from lib.mpvariable import MPVariable

def start_receiver(in_con, q, mp_drop_frames):
    """
    START THREAD
    """
    t = threading.Thread(target=receive, args=(in_con, q, mp_drop_frames,))
    t.setDaemon(True)
    t.start()
    return t

def receive(in_con, q, mp_drop_frames):
    """
    READ CONNECTION TO QUEUE
    """
    while True:
        data = in_con.recv()
        if data is None:
            q.put(data)
            break
        if q.empty():
            q.put(data)
        else:
            mp_drop_frames.value += 1
    in_con.close()
    return

def start_sender(out_con, q):
    """
    START THREAD
    """
    t = threading.Thread(target=send, args=(out_con, q, ))
    t.setDaemon(True)
    t.start()
    return t

def send(out_con, q):
    """
    READ QUEUE TO CONNECTION
    """
    while True:
        in_time = time.time()
        data = q.get(block=True)
        q.task_done()
        out_con.send(data)
        out_time = time.time()
        MPVariable.send_proc_time.value = out_time - in_time
        if data is None:
            break
    out_con.close()
    return
