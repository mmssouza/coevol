import numpy as np
from scipy.spatial.distance import pdist,squareform,euclidean
#import distances

def silhouette(X, cIDX):
    """
    Computes the silhouette score for each instance of a clustered dataset,
    which is defined as:
        s(i) = (b(i)-a(i)) / max{a(i),b(i)}
    with:
        -1 <= s(i) <= 1

    Args:
        X    : A M-by-N array of M observations in N dimensions
        cIDX : array of len M containing cluster indices (starting from zero)

    Returns:
        s    : silhouette value of each observation
    """

    N = X.shape[0]              # number of instances
    K = len(np.unique(cIDX))    # number of clusters

    # compute pairwise distance matrix
    D = squareform(pdist(X))

    # indices belonging to each cluster
    kIndices = [np.flatnonzero(cIDX==k) for k in range(K)]

    # compute a,b,s for each instance
    a = np.zeros(N)
    b = np.zeros(N)
    for i in range(N):
        # instances in same cluster other than instance itself
        a[i] = np.mean( [D[i][ind] for ind in kIndices[cIDX[i]] if ind!=i] )
        # instances in other clusters, one cluster at a time
        b[i] = np.min( [np.mean(D[i][ind]) 
                        for k,ind in enumerate(kIndices) if cIDX[i]!=k] )
    s = (b-a)/np.maximum(a,b)

    return s

def delta(A,B,dist):
  Na = A.shape[0]
  Nb = B.shape[0]
  d = squareform(pdist(np.vstack((A,B)),metric = dist))
  return d[Na:Na+Nb,0:Na].min()

def Delta(A,dist):
   return pdist(A,metric=dist).max() 
 
# Dunn measure
def di(X,cIDX,distance = 'euclidean'):
  Nclusters = cIDX.max()
  # Encontra o cluster com maior espalhamento
  aux = np.array([Delta(X[np.where(cIDX == i)],distance) for i in range(1,Nclusters+1)]).max()
  aux2 = []
  aux3 = []
  for i in range(1,Nclusters+1):
   for j in range(1,Nclusters+1):
     if i != j:
      aux2.append(delta(X[np.where(cIDX == i)],X[np.where(cIDX == j)],distance)/aux)    
   aux3.append(np.array(aux2).min())     
  return np.array(aux3).min()

# David-Bouldin's measure (db)

def db(X,cIDX,q = 1,t = 2,distance = 'euclidean'):
 Nclusters = cIDX.max()+1
# Clusters
 A = np.array([ X[np.where(cIDX == i)] for i in range(Nclusters)])
# Centroids
 v = np.array([ np.sum(Ai,axis = 0)/float(Ai.shape[0])  for Ai in A])
 
 s = []
 for Ai,vi in zip(A,v):
  s.append((np.array([euclidean(x,vi)**float(q)  for x in Ai]).sum()/float(Ai.shape[0]))**(1/float(q))) 
 
 d = squareform(pdist(v,'minkowski',t))
 
 R = []
 for i in range(Nclusters):
   R.append(np.array([(s[i] + s[j])/d[i,j] for j in range(Nclusters) if j != i]).max())
 
 return np.array(R).sum()/float(Nclusters)

# CS index : ratio of the sum of within-cluster scatter to between cluster separation.
# small CS => valid optimal partition

def CS(X,cIDX,dist='euclidean'):
#def CS(X,cIDX,dist=distances.chernoff):
 Nclusters = cIDX.max()+1
# Clusters
 A = np.array([ X[np.where(cIDX == i)] for i in range(Nclusters)])
# Centroids
 v = np.array([ np.sum(Ai,axis = 0)/float(Ai.shape[0])  for Ai in A]) 
 dv = squareform(pdist(v, metric = dist)) 
 aux1 = []
 aux2 = []
 for i in range(Nclusters):
  aux1.append(np.array([a.max() for a in pdist(A[i],metric = dist)]).sum()/float(A[i].shape[0]))
  aux2.append(dv[i,dv[i].argsort()[1]])

 return(np.array(aux1).sum()/np.array(aux2).sum())

# MM : Membership matrix
# MM shape is Nclusters x Npoints
# MM[i,j] is the membership degree of data point j to cluster i
def MM(X,cIDX, m = 2):
 Nclusters  = cIDX.max()+1
 Npoints = X.shape[0]

 M = np.ndarray(shape = (Nclusters,Npoints),dtype = float)

# Clusters
 A = np.array([ X[np.where(cIDX == i)] for i in range(Nclusters)])
# Centroids
 v = np.array([ np.sum(Ai,axis = 0)/float(Ai.shape[0])  for Ai in A]) 

 for i in range(Nclusters):
  for j in range(Npoints):
    M[i,j] = 1/np.array([(euclidean(X[j],v[i])/euclidean(X[j],v[k]))**(2/(float(m)-1)) for k in range(Nclusters)]).sum() 

 return M

# PC : Partition coefficient : Measures the amount of overlap between clusters. PC = 1 for good partition
def PC(X,cIDX):
  return (MM(X,cIDX)**2).sum()/float(X.shape[0])

# CE: Classification entropy: good partition -> minimum entropy
def CE(X,cIDX):
 M = MM(X,cIDX)
 return -(M*np.log(M)).sum()/float(X.shape[0])
 
def sm(X,cIDX,dist='euclidean'):
 Nclusters = cIDX.max()+1
 Npoints=len(X)
 
# Clusters
 A = np.array([ X[np.where(cIDX == i)] for i in range(Nclusters)])
# Centroids
 v = np.array([ np.sum(Ai,axis = 0)/float(Ai.shape[0])  for Ai in A]) 
 dv = squareform(pdist(v, metric = dist)) 

 aux1 = []
 for i in range(Nclusters):
  aux1.append((dv[i,dv[i].argsort()[1]])**2)

 M=MM(X,cIDX)
 
 z = np.ndarray(shape = (Nclusters,Npoints),dtype = float)
 for i in range(Nclusters):
  for j in range(Npoints):
   z[i,j] = (euclidean(X[j],v[i])**2)*(M[i,j]**2)
 return(z.sum()/(Npoints*np.array(aux1).min()))
 
def ch(X,cIDX,dist='euclidean'):
 Nclusters = cIDX.max()+1
 Npoints=len(X)

 n = np.ndarray(shape = (Nclusters),dtype = float)

 j=0
 for i in range(cIDX.min(),cIDX.max()+1):
  aux=np.asarray([float(b) for b in (cIDX==i)])  
  n[j]=aux.sum()
  j=j+1


# Clusters
 A = np.array([ X[np.where(cIDX == i)] for i in range(Nclusters)])
# Centroids
 v = np.array([ np.sum(Ai,axis = 0)/float(Ai.shape[0])  for Ai in A])

 ssb=0

 for i in range(Nclusters):
  ssb=n[i]*(euclidean(v[i],np.mean(X,axis=0))**2)+ssb

 z = np.ndarray(shape = (Nclusters),dtype = float)

 for i in range(cIDX.min(),cIDX.max()+1):
  aux=np.array([(euclidean(x,v[i])**2) for x in X[cIDX==i]])
  z[i]=aux.sum()
   
 ssw=z.sum()

 return((ssb/(Nclusters-1))/(ssw/(Npoints-Nclusters)))