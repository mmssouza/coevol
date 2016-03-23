import rpyc
import sys
from threading import Thread,Event
import oct2py
import numpy as np

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
sys.path.append("c:\\Users\\marce\\prj\\coevol")
	
class MyService(rpyc.Service):
    	 def on_connect(self):
           pass

         def on_disconnect(self):
          pass
         
         
         class exposed_EvalContours(object):   
            def __init__(self,names,nc = 256):
                        self.names = names
                        self.active = Event()
                        self.active.clear()
                        self.stop = False 
                        self.nc = nc
                        self.thread = Thread(target = self.work)
			self.thread.start()

            def exposed_set_nc(self,value):
             self.nc = value
 
            def exposed_active(self): 
             self.active.set()
 
            def exposed_is_busy(self):
              return self.active.is_set()

            def exposed_stop(self):
              self.stop = True
              self.active.set()
              self.thread.join()

            def work(self):
             oc = oct2py.Oct2Py(temp_dir = "c:\\users\\marce\\prj\\coevol\\tmp") 
             while True:
              self.active.wait()
              if self.stop == True: 
               oc.exit()    
               return  
         
              contours = []
              for k in self.names: 
  	           im = oc.imread("c:\\Users\\marce\\prj\\leaves_png\\"+k)
  	           s = oc.extract_longest_cont(im,self.nc)
  	           contours.append(np.array([complex(i[0],i[1]) for i in s]))
              self.exposed_contours = contours
              self.active.clear() 
			 
                        

	                
if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(MyService, hostname = "localhost", port = 18871).start()
