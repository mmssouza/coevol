import rpyc
import sys
from threading import Thread,Event
from multiprocessing import Pool
import oct2py
import numpy as np
from functools import partial

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
sys.path.append("/home/isaura/prj/coevol")


#def f(n,nc):
#          oc = oct2py.Oct2Py()
#          im = oc.imread("../leaves_png/"+n)
#          s = oc.extract_longest_cont(im,nc)
#          oc.exit()
#          return np.array([complex(i[0],i[1]) for i in s]) 

def f(n,nc):
          oc = oct2py.Oct2Py()
          oc.push(['name','nc'],["../leaves_png/"+n,nc])
          oc.eval(["im = imread(name);","c = extract_longest_cont(im,nc);"],log = False)
          s = oc.pull("c")
          oc.exit()
          return np.array([complex(i[0],i[1]) for i in s]) 

class MyService(rpyc.Service):
    	 def on_connect(self):
           self.pool = Pool(processes=4)         
           
         def on_disconnect(self):
           self.pool.close()

         def exposed_evalcontours(self,n,nc):
           contours = self.pool.map(partial(f,nc = nc),n)
           return contours
                   

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

            def exposed_set_nc(self,nc):
             self.nc = nc
 
            def exposed_is_busy(self):
              return self.active.is_set()

            def exposed_stop(self):
              self.stop = True
              self.active.set()
              self.thread.join()

            def work(self):
             oc = oct2py.Oct2Py() 
             while True:
              self.active.wait()
              if self.stop == True: 
               oc.exit()    
               return  
         
              contours = []
              for k in self.names: 
  	           im = oc.imread("../leaves_png/"+k)
  	           s = oc.extract_longest_cont(im,self.nc)
  	           contours.append(np.array([complex(i[0],i[1]) for i in s]))
              self.exposed_contours = contours
              self.active.clear() 
			 
                        

	                
if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(MyService, hostname = "localhost", port = 18871).start()
