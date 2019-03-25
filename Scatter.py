#!/usr/bin/python3
import sys
import pickle
import numpy
import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from PIL import ImageColor
from sklearn.manifold import Isomap,MDS,SpectralEmbedding
from sklearn.preprocessing import scale
import descritores as desc
#from scipy.spatial.distance import pdist

############################
#kimia99
############################
# weight = 1,0,0
# SI = 0.256 DB = 1.53 CH = 90.8
#sigma = numpy.array([ 3.16,   0.894,  6.28])

# SI = 0.297 DB = 1.07 CH = 37.7
#sigma = numpy.array([0.36511174,   2.15106849,  64.96898439])
#sigma = numpy.array([0.315, 81.2, 2.07, 5.19])
sigma = numpy.array([6.17,2.22,0.614,0.468])
############## grupo 1 ###############

# SI = 0.158 DB= 1.74 CH= 15.9
#sigma = numpy.array([2.38, 32.3, 66.6, 66.8])

# SI = 0.159  DB = 1.5  CH= 13.0
#sigma = numpy.array([2.73, 57.1, 59.6, 59.5])

# SI = 0.191  DB = 1.54  CH= 13.3
#sigma = numpy.array([0.637, 42.2, 51.3, 70.8])

# SI = 0.148  DB = 1.58  CH= 13.9
#sigma = numpy.array([0.68, 36.5, 37.8, 41.7])

############## grupo 2 ###############

# SI = 0.208  DB = 1.27  CH= 24.3
#sigma = numpy.array([0.963, 1.88, 37.3, 35.6])

# SI = 0.269  DB = 1.2  CH= 23.8
#sigma = numpy.array([0.727, 1.98, 42.7, 69.7])

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([1.21, 1.9, 64.9, 66.4])

# SI = 0.266  DB = 1.27  CH= 23.1
#sigma = numpy.array([2.31, 5.58, 65.2, 65.3])

############## grupo 3 ###############

# SI =0.284  DB=1.32  CH=97.3
#sigma = numpy.array([1.11, 1.9, 3.31, 5.29])

# SI = 0.3  DB = 1.12  CH= 37.0
#sigma = numpy.array([0.516, 3.05, 7.66, 62.1])

########################################################

#################### weights = 0.0 1.0 0.0 ########################

############## grupo 1 ###############

# SI = 0.185  DB = 1.38  CH= 13.4
#sigma = numpy.array([0.808, 59.2, 59.4, 59.8])

# SI = 0.16  DB = 1.36  CH= 12.8
#sigma = numpy.array([1.65, 58.3, 58.6, 58.6])

# SI = 0.141  DB = 1.33  CH= 14.5
#sigma = numpy.array([1.65, 62.1, 65.5, 79.9])

# SI = 0.162  DB = 1.33  CH= 14.0
#sigma = numpy.array([1.61, 61.0, 65.0, 65.1])

# SI = 0.206  DB = 1.38  CH= 13.3
#sigma = numpy.array([0.535, 58.7, 59.2, 59.4])

############## grupo 2 ###############

# SI = 0.206  DB = 1.06  CH= 22.7
#sigma = numpy.array([0.901, 7.16, 79.4, 79.6])

# SI = 0.22  DB = 1.02  CH= 23.2
#sigma = numpy.array([0.798, 6.66, 79.0, 79.9])

# SI = 0.227  DB = 1.01  CH= 23.2
#sigma = numpy.array([0.604, 6.87, 79.8, 79.9])

# SI = 0.228  DB = 1.03  CH= 23.7
#sigma = numpy.array([0.761, 6.18, 75.4, 79.8])

# SI = 0.25  DB = 1.1  CH= 22.5
#sigma = numpy.array([0.678, 6.27, 37.2, 65.0])
#########################################################

############## weights = 0.0 0.0 1.0 ###################

############## grupo 1 ###############

# SI = 0.159  DB = 1.29  CH= 51.7
#sigma = numpy.array([0.613, 0.671, 0.718, 78.6])

# SI = 0.161  DB = 1.28  CH= 52.2
#sigma = numpy.array([0.543, 0.586, 0.615, 78.6])

# SI = 0.171  DB = 1.26  CH= 51.1
#sigma = numpy.array([ 0.505, 0.539, 0.616, 72.6])

# SI = 0.159  DB = 1.29  CH= 51.1
#sigma = numpy.array([0.513, 0.704, 78.5, 0.513])

############## grupo 2 ###############

# SI = 0.155  DB = 1.33  CH= 26.6
#sigma = numpy.array([ 0.598, 0.845, 72.8, 73.1])

# SI = 0.161  DB = 1.32  CH= 26.6
#sigma = numpy.array([0.544, 0.756, 72.7, 72.7])

# SI = 0.152  DB = 1.33  CH= 26.7
#sigma = numpy.array([0.675, 0.719, 72.7, 78.7])

# SI = 0.278  DB = 1.1  CH= 26.9
#sigma = numpy.array([0.558, 2.16, 72.6, 78.6])

############## grupo 3 ###############

# SI = 0.107  DB = 1.48  CH= 16.3
#sigma = numpy.array([0.87, 72.7, 78.5, 78.6])

# SI = 0.128  DB = 1.43  CH= 16.1
#sigma = numpy.array([0.721, 72.7, 78.6, 78.7])

# SI = 0.132  DB = 1.43  CH= 16.3
#sigma = numpy.array([0.502, 73.7, 78.6, 78.7])

##########################################################

#0.0 1.0 1.0

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.267 1.07 27.1 2.92 78.6 2.24 0.857 78.5

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.269 1.08 26.2 2.99 2.25 78.7 75.9 0.884

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.27 1.06 50.3 2.05 0.679 1.94 78.6 0.526

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.143 1.41 16.4 4.46 68.3 78.6 0.539 78.6

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.141 1.41 16.3 4.48 78.6 68.0 0.636 79.9

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.278 1.06 51.5 2.03 78.6 2.05 0.508 0.777

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.273 1.07 26.9 2.93 78.6 2.15 0.501 78.8

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.281 1.06 49.6 2.07 0.707 0.735 73.0 1.99

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.277 1.06 49.4 2.07 1.89 72.5 0.513 0.532

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.283 1.07 48.7 2.09 0.831 2.04 1.01 68.6

# SI = 0.286  DB = 1.07  CH= 50.0
#sigma = numpy.array([0.533, 0.73, 2.1, 72.9])

#1.0 1.0 0.0

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.264 1.1 22.6 2.73 1.41 65.4 65.2 5.19

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.269 1.11 22.5 2.7 5.41 65.1 65.2 1.56

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.2 1.38 14.7 3.28 64.9 61.9 0.624 65.1

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.263 1.1 23.2 2.71 67.4 1.5 68.0 5.13

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.277 1.15 22.4 2.81 57.7 0.588 56.5 1.98

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.244 1.18 21.8 2.95 37.8 62.8 4.9 1.35

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.268 1.1 22.5 2.73 1.47 65.1 65.2 5.35

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.326 1.05 42.9 2.61 6.44 0.507 73.1 2.51

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.195 1.36 15.0 3.39 75.7 36.9 65.0 0.826

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.204 1.41 14.9 3.43 0.777 65.1 36.6 64.9

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.158 1.37 13.8 3.27 64.9 61.2 1.58 62.5

#1.0 0.0 1.0

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.279 1.14 25.8 3.58 67.4 2.29 68.1 0.984

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.331 1.06 44.3 2.71 67.8 2.39 5.36 0.795

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.306 1.05 43.2 2.67 1.17 70.7 2.15 5.34

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.281 1.15 26.1 3.6 2.33 69.4 0.594 68.5

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.15 1.44 16.4 5.09 0.631 68.3 72.2 78.6

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.25 1.14 24.9 3.68 67.5 1.43 2.39 69.0

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.175 1.47 15.9 5.12 68.8 0.62 68.7 68.8

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.289 1.14 26.1 3.58 2.13 0.841 67.9 70.7

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.294 1.07 49.8 2.68 0.577 2.13 68.0 0.652

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.271 1.08 46.7 2.71 68.3 1.27 0.7 1.93

# SI = 0.264  DB = 1.13  CH= 24.3
#sigma = numpy.array([64.9, 66.4, 1.9, 1.21])

#69 0.294 1.14 26.0 3.55 67.6 67.4 2.16 0.54

colors = {1:"#555500",2:"#7faa00",3:"#aaff00",4:"#ff5500",5:"#00aa2a",6:"#2aff2a",7:"#7f002a",
8:"#aa2a2a",9:"#d4552a",10:"#ff7f2a",11:"#00d455"}

#colors = {1:"#555500",2:"#7faa00",3:"#aaff00",4:"#ff5500",5:"#00aa2a",6:"#2aff2a",7:"#7f002a",
#8:"#aa2a2a",9:"#d4552a",10:"#ff7f2a",11:"#00d455",12:"#2aff55",13:"#7f2a55",14:"#aa5555",15:"#d47f55",
#16:"#ffaa55",17:"#2a2a00",18:"#2aff7f",19:"#7f007f",20:"#aa557f",21:"#d47f7f",22:"#ffaa7f",23:"#00ffaa",
#24:"#5500aa",25:"#7f2aaa",26:"#aa55aa",27:"#d4aaaa",28:"#ffd4aa",29:"#2a00d4",30:"#552ad4",31:"#7f55d4",
#32:"#aa7fd4"}

path = sys.argv[1]

with open(path+"classes.txt","rb") as f:
 with open(path+"names.pkl","rb") as g:
   cl = pickle.load(f)
   nomes = pickle.load(g)

db = {}

for im_file in nomes:
 nmbe = desc.bendenergy(path+im_file,sigma)
 db[im_file] = numpy.hstack((cl[im_file],numpy.log(nmbe())))

# nome das figuras
data1 = numpy.array([db[i] for i in db.keys()])

Y = data1[:,0].astype(int)
X1 = scale(data1[:,1:])
#s = silhouette.silhouette(X1,Y-1)
#print numpy.median(numpy.abs(1.-  s))

#iso = Isomap(n_neighbors=98, max_iter= 2500)
#mds =  MDS(n_init = 200,dissimilarity = 'euclidean',max_iter = 1500)
emb = SpectralEmbedding(n_components=2)
#X1 = iso.fit_transform(data1[:,1:])
#X1 = mds.fit_transform(data1[:,1:])
X1 = emb.fit_transform(data1[:,1:])
#r = ((pdist(data1[:,1:]) - pdist(X1))**2).sum()
#s = ((pdist(X1)-pdist(X1).mean())**2).sum()
#R2 = 1-r/s
#print R2
data = numpy.vstack((Y,X1.transpose())).transpose()

db = dict(zip(db.keys(),data))

fig = PLT.gcf()
fig.clf()
ax = PLT.subplot(111)
PLT.gray()
PLT.xlim((-3.5,3.5))
PLT.ylim((-3.5,4))
for im in db.keys():
 # add a first image
 img = Image.open(path+im)
 img.thumbnail((150,150),Image.ANTIALIAS)
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
 imagebox = OffsetImage(numpy.array(img), zoom=.15)
 xy = [db[im][1],db[im][2]]               # coordinates to position this image
 ab = AnnotationBbox(imagebox, xy,
      xybox=(5., -5.),
      xycoords='data',
      boxcoords="offset points",
	  frameon = False)
 ax.add_artist(ab)

# rest is just standard matplotlib boilerplate
ax.grid(False)
PLT.draw()
PLT.show()
