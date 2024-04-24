<h1>Python Implementation of ANDClust Clustering Algorithm</h1>
<br><br>
Although density-based clustering algorithms can successfully define clusters in arbitrary shapes, they encounter issues if the dataset has varying densities or neck-typed clusters due to the requirement for precise distance parameters, such as eps parameter of DBSCAN. These approaches assume that data density is homogenous, but this is rarely the case in practice. In this study, a new clustering algorithm named ANDClust (Adaptive NeighborhoodDistance-based Clustering Algorithm) is proposed to handle datasets with varying density and/or neck-typed clusters. The algorithm consists of three parts. The first part uses Multivariate Kernel Density Estimation (MulKDE) to find the dataset’s peak points, which are the start points for the MinimumSpanning Tree (MST) to construct clusters in the second part. Lastly, anAdaptive Neighborhood Distance (AND) ratio is used to weigh the distance between the data pairs. This method enables this approach to support inter-cluster and intra-cluster density varieties by acting as if the distance parameter differs for each data of the dataset. ANDClust is tested on synthetic and real datasets to reveal its efficiency. The algorithm shows superior clustering quality in a good run-time compared to its competitors. Moreover,ANDClust could effectively define clusters of arbitrary shapes and process high-dimensional, imbalanced datasets may have outliers.<br>
<br>
The main contributions of our algorithm can be summed up as follows:

• Since the proposed algorithm is density-based, it can define arbitrary-shaped clusters, 

• Thanks to the flexible neighborhood distance approach, it can handle not only the varying density among clusters but also the varying density inside the cluster,

• ANDClust is robust against outliers/noisy data,

• It can handle high-dimensional data,

• It can handle imbalanced datasets,

• Its clustering quality is high,

• It can handle datasets that have neck-typed clusters.


![Micro-Clusters](img/1_HalfKernel_.png) 
![Macro-Clusters](img/1_HalfKernel__ARI.png)

![Micro-Clusters](img/2_Three_Spirals_.png) 
![Macro-Clusters](img/2_Three_Spirals__ARI.png)

![Micro-Clusters](img/3_Corners_.png) 
![Macro-Clusters](img/3_Corners__ARI.png)

![Micro-Clusters](img/4_Moon_.png) 
![Macro-Clusters](img/4_Moon__ARI.png)

Requirements:

• Python (2.9 or upper)

• NumPy

• scikit-learn (for KDTree, MinMaxScaler, and some metrics)

• SciPy (optional, for loading .mat files)

• Seaborn (for plotting)

• Matplotlib (for creating plots)

• IPython (for some IPython-specific magic commands)

• Datasets (Place your datasets in a directory named "Datasets")

• Directory Structure (Create an "img" directory in the same location as your code)

• Jupyter Notebook (optional, if you plan to run the code in a Jupyter Notebook)

If you use the code in your works, please cite the paper given below:
<br>
<h2>How to Cite:</h2><br>
A. Şenol, ANDClust: An Adaptive Neighborhood Distance-Based Clustering Algorithm to Cluster Varying Density and/or Neck-Typed Datasets. Adv. Theory Simul. 2024, 7, 2301113. https://doi.org/10.1002/adts.202301113
