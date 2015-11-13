#!/usr/bin/python
import sys
import cPickle
import numpy
import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from PIL import ImageColor
from sklearn.manifold import Isomap,MDS
from sklearn.preprocessing import scale
import descritores as desc
from scipy.spatial.distance import pdist
import silhouette

############################
#kimia99
############################

# Fitness = 1.05
#sigma = numpy.array([78.21, 61.27, 53.77, 17.76,  10.73 , 2.02])

# Fitness = 0.85

#sigma = numpy.array([112.91,80.36,74.13,72.07,49.63,45.45,38.30])

# Fitness = 0.708
#sigma = numpy.array([65.,6.407,2.351,0.567,0.523,0.442,0.384])

# Fitness = 0.698
#sigma = numpy.array([2.21,2.34,0.8,0.86,6.1,31.,5.85])

# Fitness = 0.695
#sigma = numpy.array([0.852,0.646,0.336,5.72,2.914,2.094,1.016])

# Fitness = 0.688
#sigma = numpy.array([62.87,6.379,5.501,2.23,2.026,0.474,0.248])

# Fitness = 0.685
#sigma = numpy.array([6.16895884,2.65783154,3.19456357,2.1800385,0.25585797,1.99563044,1.9639186])

#Fitness = 0.6844
#sigma = numpy.array([65.0,6.5,5.3,2.4,2.1,1.02])  

#Fitness = 0.684
#sigma = numpy.array([ 2.21457565,  2.40416633,  3.01947702,  5.45400147,  5.75948417,  0.36157774, 0.62346497])

# Fitness = 0.680
#sigma = numpy.array([65.08,6.7,6.3,2.4,2.2,2.1,0.51])

# Fitness = 0.669486503726
#sigma = numpy.array([65.1,7.1,5.95,2.6,2.3,0.68,0.32])

# Fitness = 0.668349234233
#sigma = numpy.array([65.07,6.7,5.9,2.5,2.4,2.2,0.54]) 

# Fitness = 0.668140653845
#sigma = numpy.array([65.07,6.39,2.63,2.35,0.7,0.29720314]) 

# MAD = 0.603
sigma = numpy.array([ 0.98257302,   55.96277287,    1.74498014,    5.58365476])

# Fitness = -0.5 (Mediana silhouette)
#sigma = numpy.array([ 2.35139906,  47.82679226,   7.09654377,  50.53723473,   0.25904659])
# Fitness = -0.53 (Mediana Silhouette)
#sigma = numpy.array([ 49.9,  0.6857,  47.13,  2.35, 7.31])
# Fitness =  -0.5053603757
#sigma = numpy.array([ 50.7,  7.5,   0.367,   2.6])

###################################################################
###################################################################
#0 0.827572521068
#sigma = numpy.array([ 118.82641744,   42.08713877,   75.30066905,   #31.59915181,   45.65483767, 34.55784572,  111.27537105])
#3 0.818466188317
#sigma = numpy.array([ 123.795453,     32.3369149,   105.40396206,   #88.76472376,   83.29176694,   41.50664413,   67.44670651])
#4 0.771112451306
#sigma = numpy.array([ 38.78997795,  59.32258068,  48.80320769,  58.24660811,  52.79860366, 18.42376106,  37.21951345])
#10 0.767322968144
#sigma = numpy.array([  65.49184459,   24.58816301,   14.52279183,   88.99300149,   56.85420297, 107.68521453,   95.23293596])
#25 0.735810548271
#sigma = numpy.array([  7.43748406,  58.33619828,  70.1737463,   39.45486639,  #66.05673865,  37.78311627,  75.75775078])
#35 0.726944060991
#sigma = numpy.array([ 122.07938329,   90.42501722,   27.74071011,   18.17565441,   57.92540365, 72.40635349,    2.74631296])
#52 0.705040462734
#sigma = numpy.array([ 15.99046588,   8.26661183,  79.65061262,   5.41076984,  32.98822156,  17.44628558,   5.8042881 ])
#81 0.699905965338
#sigma = numpy.array([ 10.2,   0.138,   5.19,   19.9,   4.08,  22.95,  86.02])
#97 0.695074497739
#sigma = numpy.array([ 28.43,   6.34,   3.49,   1.55,   8.37, 76.7,   5.82])
#106 0.692984064141
#sigma = numpy.array([  2.41,  11.46,  59.58,  10.037,   5.95, 3.04,   9.989])
#113 0.692813594912
#sigma = numpy.array([ 27.67,   3.76,  79.02,   1.63,   2.36,  9.92,  2.20])
#144 0.69114509391
#sigma = numpy.array([2.38,   6.75,  10.12,   0.189,  16.34,78.53,   1.55])

##########################
#leaves
##########################
#sigma = scipy.array([68.5,120.9,4.88,100.64,0.505,2.05,122.3])

#sigma = numpy.array([82.2,0.86,3.8,30.0,0.43,0.56,2.2,123.1])

#sigma = numpy.array([3.6,1.1,93.4,0.57,123.9,0.39,0.78,102.8,32.8])

#sigma = numpy.array([30.6,0.71,0.85,122.9,0.80,91.2,0.5,109.8,4.2,120.0])

#sigma = numpy.array([84.0,0.77,0.56,114.6,1.22,4.0,0.71,124.7,1.45,30.8])

#sigma = numpy.array([8.69990685, 2.96620678, 124.12750119, 4.2851001 ,
  #     0.19552142, 0.17752748, 1.91003054,  3.12253972,
    #   2.6842843 , 67.80282138, 1.00836474, 0.23692545,
      # 3.62541098, 0.15149804,  120.49299263])

#  mean silhouette = -0.143  
#sigma = numpy.array([   1.61253252,   11.43612451,    0.58806737,  109.87859876,    2.88713735])

#mean silhouette = -0.132
#sigma = numpy.array([   2.24420614,    9.82516481,  124.83165341,  122.50265062,
          #1.53460893])

# MAD = 0.923 R2 = 0.97
#sigma = numpy.array([  90.39356028,   44.49638044 ,  43.51058475 , 121.24898221,    4.68126646,
#  112.01345512,   43.70676835,    1.19320202,   43.38704464,   28.90928878,
 # 119.4322081,    85.75757847,   93.50026504,   57.0532458,     1.13643642,
  #113.25991824,   47.11634067,  123.49362002,    1.33119089 , 125.12442583])

#sigma =numpy.random.rand(11)*125 + 0.125

# MAD = 0.931 R2 = 0.98
#sigma = numpy.array([125.12442583,  121.24898221, 113.25991824,   93.50026504,   85.75757847,   57.0532458, 47.11634067,   43.51058475 ,     28.90928878,  4.68126646,  1.33119089])
	   
# MAD = 0.839 R2 = 0.99
#sigma = numpy.array([124.16695613,   93.37727823,  31.69402865 ,  3.51881064,  2.14215328,   0.95603169,    0.7488448,  0.3641014])   
 
# MAD = 0.853
#sigma = numpy.array([ 46.0,  37.03,  23.58,  17.25,   1.924 ])

#MAD = 0.872
#sigma = numpy.array([ 41.21,  34.96, 19.06,  14.52, 1.821 ])

#MAD = 0.8412
# sigma = numpy.array([ 40.684,  29.580,  2.669,   1.292 , 0.163 ])
 
#MAD = 0.806112370165
#sigma = numpy.array([   5.61336454,    1.94508667,    0.83223209,    0.29260598,   81.76772573,
#   54.9519924,    67.36983136,  104.71769155])

#MAD = 0.77847301168514837
#sigma = numpy.array([  53.03032859,    0.80239139,    1.15188946, 1.55907899,
#          0.15173929,    8.34292094,    0.34385214,  108.16470865])
		  
colors = {1:"#555500",2:"#7faa00",3:"#aaff00",4:"#ff5500",5:"#00aa2a",6:"#2aff2a",7:"#7f002a",	
8:"#aa2a2a",9:"#d4552a",10:"#ff7f2a",11:"#00d455",12:"#2aff55",13:"#7f2a55",14:"#aa5555",15:"#d47f55",	
16:"#ffaa55",17:"#2a2a00",18:"#2aff7f",19:"#7f007f",20:"#aa557f",21:"#d47f7f",22:"#ffaa7f",23:"#00ffaa",	
24:"#5500aa",25:"#7f2aaa",26:"#aa55aa",27:"#d4aaaa",28:"#ffd4aa",29:"#2a00d4",30:"#552ad4",31:"#7f55d4",	
32:"#aa7fd4"}

path = sys.argv[1]

with open(path+"classes.txt","r") as f:
 with open(path+"names.pkl","r") as g:
   cl = cPickle.load(f)
   nomes = cPickle.load(g)

db = {}

for im_file in nomes:
 nmbe = desc.bendenergy(path+im_file,sigma)
 db[im_file] = numpy.hstack((cl[im_file],numpy.log(nmbe())))

# nome das figuras
data1 = numpy.array([db[i] for i in db.keys()])

Y = data1[:,0].astype(int)
X1 = scale(data1[:,1:])
s = silhouette.silhouette(X1,Y-1)
print numpy.median(numpy.abs(1.-  s))

#iso = Isomap(n_neighbors=98, max_iter= 2500)
mds =  MDS(n_init = 20,dissimilarity = 'euclidean',max_iter = 2500)
#X1 = iso.fit_transform(data1[:,1:])
X1 = mds.fit_transform(data1[:,1:])

r = ((pdist(data1[:,1:]) - pdist(X1))**2).sum()
s = ((pdist(X1)-pdist(X1).mean())**2).sum()
R2 = 1-r/s
print R2
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

