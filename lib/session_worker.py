# -*- coding: utf-8 -*-
# TensorFlow Session Thread
#
# usage:
# before:
#     results = sess.run([opt1,opt2],feed_dict={input_x:x,input_y:y})
# after:
#     opts = [opt1,opt2]
#     feeds = {input_x:x,input_y:y}
#     woker = SessionWorker("TAG",graph,config)
#     worker.put_sess_queue(opts,feeds)
#     q = worker.get_result_queue()
#     if q is None:
#         continue
#     results = q['results']
#     extras = q['extras']
#
# extras: None or frame image data for draw. GPU detection thread doesn't wait result. Therefore, keep frame image data if you want to draw detection result boxes on image.
#
import threading
import time
import tensorflow as tf

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue

class SessionWorker():
    def __init__(self,tag,graph,config):
        self.lock = threading.Lock()
        self.sess_queue = Queue.Queue()
        self.result_queue = Queue.Queue()
        self.tag = tag
        t = threading.Thread(target=self.execution,args=(graph,config))
        t.setDaemon(True)
        t.start()
        return

    def execution(self,graph,config):
        self.is_thread_running = True
        try:
            start_time,start_clock = time.time(),time.clock()
            with tf.Session(graph=graph,config=config) as sess:
                end_time,end_clock=time.time()-start_time,time.clock()-start_clock
                print("{} - sess time:{:.8f},clock:{:.8f}".format(self.tag,end_time,end_clock))
                while self.is_thread_running:
                    #with self.lock:
                        while not self.sess_queue.empty():
                            start_time,start_clock=time.time(),time.clock()
                            q = self.sess_queue.get(block=False)
                            opts = q["opts"]
                            feeds= q["feeds"]
                            extras= q["extras"]

                            if feeds is None:
                                results = sess.run(opts)
                            else:
                                results = sess.run(opts,feed_dict=feeds)
                            self.result_queue.put({"results":results,"extras":extras})
                            self.sess_queue.task_done()
                            end_time,end_clock=time.time()-start_time,time.clock()-start_clock
                            print("{} - time:{:.8f},clock:{:.8f}".format(self.tag,end_time,end_clock))
                        time.sleep(0.005)
                    # 実行するqueueが空の時はsleepする
                    #time.sleep(0.005)
        except:
            import traceback
            traceback.print_exc()
        self.stop()
        print("end execution")
        return

    def is_sess_empty(self):
        if self.sess_queue.empty():
            return True
        else:
            return False

    def put_sess_queue(self,opts,feeds=None,extras=None):
        '''
        新しい処理をqueueに入れる
        #処理中は次のqueueが入らないようにlockする
        '''
        start_time,start_clock=time.time(),time.clock()
        #with self.lock:
        self.sess_queue.put({"opts":opts,"feeds":feeds,"extras":extras})
        end_time,end_clock=time.time()-start_time,time.clock()-start_clock
        print("{} put_sess_queue(), time:{:.8f} clock:{:.8f}".format(self.tag,end_time,end_clock))
        return

    def is_result_empty(self):
        if self.result_queue.empty():
            return True
        else:
            return False

    def get_result_queue(self):
        start_time,start_clock=time.time(),time.clock()
        result = None
        if not self.result_queue.empty():
            result = self.result_queue.get(block=False)
            self.result_queue.task_done()
        end_time,end_clock=time.time()-start_time,time.clock()-start_clock
        print("{} get_result_queue(), time:{:.8f} clock:{:.8f}".format(self.tag,end_time,end_clock))
        return result

    def stop(self):
        self.is_thread_running=False
        with self.lock:
            while not self.sess_queue.empty():
                q = self.sess_queue.get(block=False)
                self.sess_queue.task_done()
        return
