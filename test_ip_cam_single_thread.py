import numpy as np
import cv2
import time
import requests
import threading
from threading import Thread, Event, ThreadError
from PIL import Image

class Cam():

  def __init__(self, url):
    self.stream = requests.get(url, stream=True)
    self.thread_cancelled = False
    print("camera initialised")
    
  def start(self):
    bytes=b''
    while not self.thread_cancelled:
      try:
        # print(self.stream.raw.read(1024))
        bytes+=self.stream.raw.read(1024) 
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        # print(a)
        if a!=-1 and b!=-1:
          jpg = bytes[a:b+2]
          bytes= bytes[b+2:]
          if not jpg == b'':
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
            # ----------For saving a image---------------
            # data = img
            # rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

            # im = Image.fromarray(rescaled)
            # im.save(f'test.png')

            
            cv2.imshow('cam',img)
            if cv2.waitKey(1) == 27:
                exit(0)
      except ThreadError:
        self.thread_cancelled = True
        
        
  def is_running(self):
    return self.thread.isAlive()
      
    
  def shut_down(self):
    self.thread_cancelled = True
    #block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True

  
    
if __name__ == "__main__":
  url = 'http://192.168.1.108:81/stream'
  cam = Cam(url)
  cam.start()