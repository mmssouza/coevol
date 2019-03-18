#!/usr/bin/python3
import sys
import pickle
import numpy
import pylab
import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from PIL import ImageColor
from sklearn.manifold import Isomap,MDS
from sklearn.preprocessing import scale
import descritores as desc
#from scipy.spatial.distance import pdist

############################
#kimia99
############################
#pareto = pylab.loadtxt(sys.argv[1])

colors = {1:"#555500",2:"#7faa00",3:"#aaff00",4:"#ff5500",5:"#00aa2a",6:"#2aff2a",7:"#7f002a",
8:"#aa2a2a",9:"#d4552a",10:"#ff7f2a",11:"#00d455"}

#colors = {1:"#555500",2:"#7faa00",3:"#aaff00",4:"#ff5500",5:"#00aa2a",6:"#2aff2a",7:"#7f002a",
#8:"#aa2a2a",9:"#d4552a",10:"#ff7f2a",11:"#00d455",12:"#2aff55",13:"#7f2a55",14:"#aa5555",15:"#d47f55",
#16:"#ffaa55",17:"#2a2a00",18:"#2aff7f",19:"#7f007f",20:"#aa557f",21:"#d47f7f",22:"#ffaa7f",23:"#00ffaa",
#24:"#5500aa",25:"#7f2aaa",26:"#aa55aa",27:"#d4aaaa",28:"#ffd4aa",29:"#2a00d4",30:"#552ad4",31:"#7f55d4",
#32:"#aa7fd4"}

dim = 5
path = sys.argv[1]

with open(path+"classes.txt","rb") as f:
 with open(path+"names.pkl","rb") as g:
   cl = pickle.load(f)
   nomes = pickle.load(g)

def scatter(ss,fig):
   db = {}
   for im_file in nomes:
       nmbe = desc.bendenergy(path+im_file,ss)
       db[im_file] = numpy.hstack((cl[im_file],numpy.log(nmbe())))
       # nome das figuras
   data1 = numpy.array([db[i] for i in db.keys()])
   Y = data1[:,0].astype(int)
   X1 = scale(data1[:,1:])
   #iso = Isomap(n_neighbors=98, max_iter= 2500)
   mds =  MDS(n_init = 20,dissimilarity = 'euclidean',max_iter = 2500)
   #X1 = iso.fit_transform(data1[:,1:])
   X1 = mds.fit_transform(data1[:,1:])

   #r = ((pdist(data1[:,1:]) - pdist(X1))**2).sum()
   #s = ((pdist(X1)-pdist(X1).mean())**2).sum()
   #R2 = 1-r/s
       #print R2
   data = numpy.vstack((Y,X1.transpose())).transpose()
   db = dict(zip(db.keys(),data))
   ax = PLT.subplot(111)
   PLT.gray()
   PLT.xlim((-5.,5.))
   PLT.ylim((-5.,5.))
   for im in db.keys():
       # add a first image
       img = Image.open(path+im)
       img.thumbnail((180,180),Image.ANTIALIAS)
       #img = PIL.ImageOps.invert(img.convert("L"))
       img = img.convert("RGBA")
       datas = img.getdata()
       newData = []
       for item in datas:
           if item[0] == 255 and item[1] == 255 and item[2] == 255:
               newData.append((255, 255, 255, 0))
           else:
               newData.append(ImageColor.getrgb(colors[int(db[im][0])]))
       img.putdata(newData)
       imagebox = OffsetImage(numpy.array(img), zoom=.2)
       xy = [db[im][1],db[im][2]]               # coordinates to position this image
       ab = AnnotationBbox(imagebox, xy,
           xybox=(5., -5.),
           xycoords='data',
           boxcoords="offset points",
           frameon = False)
       ax.add_artist(ab)
       ax.grid(False)

fig = PLT.gcf()
fig.clf()
multi = pylab.array([[5.38,2.36,.818],[85.3,5.925,0.468],[83.9,2.,0.954]])
si = pylab.array([[80.85,4.04,1.091],[5.865,2.285,0.413]])
db = pylab.array([[81.4,6.39,.436]])
ch = pylab.array([[2.3,0.788,0.379]])
tt = pylab.array([[0.315,81.2,2.07,5.19]])
aux = tt
idx = 0
#st1 = "si = {:2.3} db = {:2.3} ch = {:2.3}".format(aux[dim],aux[dim+1],aux[dim+2])
st2 = str().join(["{:2.3} ".format(i) for i in aux[idx]])
PLT.title("Scales: "+st2)
scatter(aux[idx],fig)
PLT.draw()
PLT.show()
#PLT.savefig("fig000.png",dpi=180,size =(15.,7.))
