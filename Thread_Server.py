import rpyc
import sys
from threading import Thread,Event
from multiprocessing import Pool
import descritores as desc
import oct2py
import numpy as np
from functools import partial

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
sys.path.append("/home/marcelo/prj/coevol")

def ft(n,s):
   tmp = desc.TAS(n,method = 'octave').sig
   tmp_h = np.histogram(tmp,bins = int(s[0]),range = (s[1],s[2]))[0].astype(float)
   return tmp_h
	
class MyService(rpyc.Service):
    	 def on_connect(self):
          self.pool = Pool(processes = 8)   

         def on_disconnect(self):
          self.pool.close() 

         def exposed_CalcTas(self,cont,args):
           res =  self.pool.map(partial(ft,s = args),cont)
           return res
              
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
              oc = oct2py.Oct2Py(executable = "/usr/bin/octave-cli") 
              while True:
               self.active.wait()
               if self.stop == True: 
                 oc.exit()    
                 return           
  	       contours = []
 	       for k in self.names: 
  	        im = oc.imread("/home/marcelo/prj/leaves_png/"+k)
  	        s = oc.extract_longest_cont(im,self.nc)
  	        contours.append(np.array([complex(i[0],i[1]) for i in s]))
 	       self.exposed_contours = contours
               self.active.clear()           

	                
if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(MyService, port = 18871).start()
