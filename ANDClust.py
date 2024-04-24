import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance_matrix
import heapq
import scipy.io
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class ANDClust:
    def __init__(self,X,N,k,eps,krnl,b_width):
        self.X=X
        self.d = X.shape[1]
        self.N=N
        self.k=k
        self.eps=eps
        self.krnl=krnl,
        self.b_width=b_width
        self.starting_point=[]
        self.distWeight=np.empty((X.shape[0],1),float)
        self.clusterNo=0
        self.labels_=np.zeros(X.shape[0]) 
        
        # self.ClusterList=np.empty((0,self.X.shape[1]+4),float)
        # self.edge_lists=[]
        
        
        self.avg_distance_knn_avg()
        self.DistWeight()
        self.multivariate_kde()
        self.ClusterDefinition()
        

    def DistWeight2(self):
        kdt = scipy.spatial.cKDTree(self.X)
        dists, neighs = kdt.query(self.X, self.k+1)
        dists[:, 0] = neighs[:, 0]
        self.distWeight = np.mean(dists[:, 1:], axis=1)
    def DistWeight(self):
        kdt = scipy.spatial.cKDTree(self.X)
        dists, neighs = kdt.query(self.X, self.k+1)
        dists[:, 0] = neighs[:, 0]
        avg_dists = np.mean(dists[:, 1:], axis=1)
    
        avg_dists2 = np.empty((0, 1), float)
        for i in range(neighs.shape[0]):
            avg_dists2 = np.vstack((avg_dists2, np.mean(avg_dists[neighs[i, 1:]])))
        self.distWeight= avg_dists2
    def avg_distance_knn_avg(self):
        # Compute the distance matrix between all pairs of data points
        distances = distance_matrix(self.X, self.X)
    
        # Find the indices of the k nearest neighbors for each data point
        knn_indices = np.argsort(distances, axis=1)[:, 1:self.k+1]
    
        # Compute the average distance of the k nearest neighbors for each data point
        knn_avg_distances = np.mean(distances[np.arange(len(self.X))[:, np.newaxis], knn_indices], axis=1)
    
        # Find the indices of the nearest k neighbors for each data point
        nearest_indices = np.argsort(distances, axis=1)[:, 1:self.k+1]
        
        # Add the distance from each data point to its nearest k neighbors to the list
        itself=np.arange(distances.shape[0])
        nearest_indices=np.hstack((itself.reshape(nearest_indices.shape[0],1),nearest_indices))
        
        # Compute the average distance of the nearest k neighbors for each data point
        self.distWeight= np.mean(knn_avg_distances[nearest_indices], axis=1)


    def multivariate_kde(self):
        # Compute the kernel density estimate for the given dataset
        kde = KernelDensity(kernel=self.krnl[0], bandwidth=self.b_width).fit(self.X)
        self.x=kde.score_samples(self.X)

    def define_cluster_by_starting_point(self):
        # compute the distance matrix
        # self.X[self.labels_!=0,:]=float('inf')
        distances = cdist(self.X, self.X) ############Çözüm bu daha önce tanımlanmış veri tekrar kullanılıyor.
        # print(distances.shape[0])
        for i in range(distances.shape[0]):
            if(self.labels_[i]!=0):
                distances[i,:]=float('inf')
                distances[:,i]=float('inf')
        # build the minimum spanning tree
        mst = []
        visited = set()
        heap = [(0, self.starting_point, -1)]
        while heap:
            (distance, current, parent) = heapq.heappop(heap)
            # print(enumerate(distances[current]))
            if current in visited or self.x[current]==np.nan:
                continue
            visited.add(current)
            if parent >= 0:
                mst.append((parent, current))
            for i, d in enumerate(distances[current]):
                if i not in visited and d/self.distWeight[current]<=self.eps+ 1 and d/self.distWeight[current]>=1-self.eps: # :# and 
                    heapq.heappush(heap, (d, i, current))
    
        # define the cluster by traversing the mst
        selected = set()
        queue = [self.starting_point]
        count=0
        while queue:
            current = queue.pop(0)
            selected.add(current)
            for (u, v) in mst:
                if u == current and v not in selected:
                    queue.append(v)
                    count=count+1
                elif v == current and u not in selected:
                    queue.append(u)
                    count=count+1
        # return the indices of the selected points
        # print("count=",count)
        # return np.array(list(selected))
        return np.array(list(selected)) if selected else None

    def ClusterDefinition(self):
        while (np.count_nonzero(~np.isnan(self.x))>=self.N):
            self.starting_point=np.where(self.x == np.nanmax(self.x))[0][0]
            # print("StartingPoint=",X[clusterCenter,:])
            # print("clusterCenter=",self.starting_point)
            cluster_indices=self.define_cluster_by_starting_point()
            # print(cluster_indices)
            self.x[self.starting_point]=np.nan
            for a in np.unique(cluster_indices): 
                self.x[a]=np.nan
            if (cluster_indices.shape[0]>=self.N):
                self.clusterNo=self.clusterNo+1
                self.labels_[cluster_indices]=self.clusterNo
                print("Cluster # %d is defined"%self.clusterNo)
                # draw_dataset_with_clusters(X, [cluster_indices])
            # else:
                # print("len(cluster_indices)=",len(cluster_indices)
    def plotGraph(self,index,index_value,dataset_name,dpi=100):
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams["figure.figsize"] = (4,4) 
        plt.scatter([self.X[self.labels_!=0, 0]], [self.X[self.labels_!=0, 1]],s=40, c=self.labels_[self.labels_!=0],edgecolors='k',cmap="jet") #nipy_spectral
        plt.scatter([self.X[self.labels_==0, 0]], [self.X[self.labels_==0, 1]],s=10, c="black",edgecolors='k',cmap="jet") #nipy_spectral
        s=str("ANDClust (N=%d, eps=%0.3f, k=%d, b_width=%0.4f,\nkernel=\'%s\') => {%s=%0.4f}"%(self.N,self.eps,self.k,self.b_width,self.krnl[0],index,index_value)) 
        
        plt.title(s,fontsize = 10,fontname="Times New Roman",fontweight="bold")
        plt.rcParams.update({'font.size': 8})
        plt.grid()
        plt.rcParams['axes.axisbelow'] = True
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt_name=str("img/"+dataset_name+"_"+index+".png")
        plt.savefig(plt_name,bbox_inches='tight') 
        plt.show()