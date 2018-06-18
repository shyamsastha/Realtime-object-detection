# -*- coding: utf-8 -*-
# TensorFlow Session Thread
#
# usage:
# before:
#     results = sess.run([opt1,opt2],feed_dict={input_x:x,input_y:y})
# after:
#     opts = [opt1,opt2]
#     feeds = {input_x:x,input_y:y}
#     extras = {"image":image} # Optional. 
#     woker = SessionWorker("TAG",graph,config)
#     worker.put_sess_queue(opts,feeds)
#     q = worker.get_result_queue()
#     if q is None:
#         continue
#     results = q['results']
#     extras = q['extras']
#     image = extras['image']
#     tag_in_time = extras['TAG_in_time']        # work in time. TAG is GPU or CPU.
#     tag_out_time = extras['TAG_out_time']      # work out time. TAG is GPU or CPU.
#     tag_proc_time = tag_out_time - tag_in_time # processing time.
#     print("TAG proc time:{}".format(tag_proc_time))
#
# extras: {} or {"image":image}. image is frame data for draw. GPU detection thread doesn't wait result. Therefore, keep frame image data if you want to draw detection result boxes on image.
#
# UPDATE:
#  - add flexible sleep_interval. This will speed up FPS on high spec machine.
#  - add in_time and out_time for processing time check
#  - change extras type to DICT.
#
# PROTOTYPE FUNCTION:
# - add secondary queue in case the CPU slow down.
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
        self.sess_queue = Queue.Queue(maxsize=1)
        self.sess_secondary_queue = Queue.Queue(maxsize=1)
        self.result_queue = Queue.Queue()
        self.tag = tag
        self.sleep_interval = 0.005
        self.jit_done = False
        t = threading.Thread(target=self.execution,args=(graph, config,))
        t.setDaemon(True)
        t.start()
        return

    def execution(self, graph, config):
        self.is_thread_running = True
        try:
            with graph.as_default():
                #start_time,start_clock = time.time(),time.clock()
                with tf.Session(config=config) as sess:
                    #end_time,end_clock=time.time()-start_time,time.clock()-start_clock
                    #print("{} - sess time:{:.8f},clock:{:.8f}".format(self.tag,end_time,end_clock))
                    while self.is_thread_running:
                        #with self.lock:
                            while not self.sess_queue.empty():
                                #start_time,start_clock=time.time(),time.clock()
                                if not self.sess_secondary_queue.empty():
                                    self.sess_queue.get(block=False)
                                    q = self.sess_secondary_queue.get(block=False)
                                    self.sess_secondary_queue.task_done()
                                else:
                                    q = self.sess_queue.get(block=False)
                                opts = q["opts"]
                                feeds= q["feeds"]
                                extras= q["extras"]
                                if extras is not None:
                                    """ add in time """
                                    extras.update({self.tag+"_in_time":time.time()})
                                if feeds is None:
                                    results = sess.run(opts)
                                else:
                                    results = sess.run(opts,feed_dict=feeds)
                                if extras is not None:
                                    """ add out time """
                                    extras.update({self.tag+"_out_time":time.time()})
                                self.result_queue.put({"results":results,"extras":extras})
                                self.sess_queue.task_done()
                                #end_time,end_clock=time.time()-start_time,time.clock()-start_clock
                                #print("{} - time:{:.8f},clock:{:.8f}".format(self.tag,end_time,end_clock))
                            time.sleep(self.sleep_interval)
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

    def is_sess_secondary_empty(self):
        if self.sess_secondary_queue.empty():
            return True
        else:
            return False

    def put_sess_queue(self,opts,feeds=None,extras=None):
        '''
        新しい処理をqueueに入れる
        #処理中は次のqueueが入らないようにlockする
        '''
        #start_time,start_clock=time.time(),time.clock()
        #with self.lock:
        self.sess_queue.put({"opts":opts,"feeds":feeds,"extras":extras})
        #end_time,end_clock=time.time()-start_time,time.clock()-start_clock
        #print("{} put_sess_queue(), time:{:.8f} clock:{:.8f}".format(self.tag,end_time,end_clock))
        return

    def put_sess_secondary_queue(self,opts,feeds=None,extras=None):
        '''
        Secondary queue.
        This will use when primary queue is full.
        '''
        self.sess_secondary_queue.put({"opts":opts,"feeds":feeds,"extras":extras})
        return

    def is_result_empty(self):
        if self.result_queue.empty():
            return True
        else:
            return False

    def get_result_queue(self):
        #start_time,start_clock=time.time(),time.clock()
        result = None
        if not self.result_queue.empty():
            result = self.result_queue.get(block=False)
            self.result_queue.task_done()
        #end_time,end_clock=time.time()-start_time,time.clock()-start_clock
        #print("{} get_result_queue(), time:{:.8f} clock:{:.8f}".format(self.tag,end_time,end_clock))
        return result

    def stop(self):
        self.is_thread_running=False
        with self.lock:
            while not self.sess_queue.empty():
                q = self.sess_queue.get(block=False)
                self.sess_queue.task_done()
        return
