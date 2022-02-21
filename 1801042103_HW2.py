from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from clustviz.chameleon.chameleon import cluster
from clustviz.chameleon.graphtools import plot2d_data
from mlxtend.frequent_patterns import fpgrowth
from matplotlib import pyplot
import pandas as pd
import time 
import sklearn
from mlxtend.preprocessing import TransactionEncoder


df, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=6)
df20D, _ = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=6)

# k-means clustering
start = time.time()
kMeansModel = KMeans(n_clusters=3)
kMeansModel.fit(df)
yhat = kMeansModel.predict(df)
clusters = unique(yhat)
end = time.time()
print("\nK-Means 2D dataset Graph\n")
for clusterx in clusters:
	row_ix = where(yhat == clusterx)
	pyplot.scatter(df[row_ix, 0], df[row_ix, 1])
pyplot.show()
print("Computational time for 2D dataset KMeans : %f" % (end-start))
print("Silhouette score for 2D dataset KMeans : %f" % (sklearn.metrics.silhouette_score(df, yhat)))

start = time.time()
kMeansModel = KMeans(n_clusters=3)
kMeansModel.fit(df20D)
yhat = kMeansModel.predict(df20D)
clusters = unique(yhat)
end = time.time()
print("Computational time for 20D dataset KMeans : %f" % (end-start))
print("Silhouette score for 20D dataset KMeans : %f" % (sklearn.metrics.silhouette_score(df20D, yhat)))


#dbscan clustering
start = time.time()
DBSCANmodel = DBSCAN(eps=0.30, min_samples=9)
yhat=DBSCANmodel.fit_predict(df)
clusters=unique(yhat)
end = time.time()
print("\nDBSCAN 2D dataset Graph\n")
for clusterx in clusters:
    row_ix = where(yhat==clusterx)
    pyplot.scatter(df[row_ix,0],df[row_ix,1])
pyplot.show()
print("Computational time for 2D dataset DBSCAN : %f" % (end-start))
print("Silhouette score for 2D dataset DBSCAN : %f\n" % (sklearn.metrics.silhouette_score(df, yhat)))

start = time.time()
DBSCANmodel = DBSCAN(eps=0.30, min_samples=9)
yhat=DBSCANmodel.fit_predict(df20D)
clusters=unique(yhat)
end = time.time()
print("Computational time for 20D dataset DBSCAN : %f" % (end-start))
#print("Silhouette score for 20D dataset DBSCAN : %f" % (sklearn.metrics.silhouette_score(dfx, yhat)))


#chameleon clustering
#k = n_clusters, knn = number of nearest neighbors, m = n_clusters to reach in the initial clustering phase, 
#alpha = exponent of relative closeness
start = time.time()
chameleonModel,h = cluster(pd.DataFrame(df),k=2,knn=15,m=10,alpha=2,plot=False)
end = time.time()
print("\nChameleon 2D dataset Graph\n")
plot2d_data(chameleonModel)
print("Computational time for 2D dataset Chameleon : %f" % (end-start))

start = time.time()
chameleonModel,h = cluster(pd.DataFrame(df20D),k=4,knn=15,m=10,alpha=2,plot=False)
end = time.time()
print("Computational time for 20D dataset Chameleon : %f\n" % (end-start))


#fpgrowth
data2D =[[1, 0],[1, 1],[0, 1],[0, 1],[0, 0],[1, 1],[1, 0],[1, 0],[0, 1],[1, 1]]
my_transactionencoder= TransactionEncoder()
my_transactionencoder.fit(data2D)
encoded_transactions = my_transactionencoder.transform(data2D)

start = time.time()
frequent_itemsets=fpgrowth(pd.DataFrame(encoded_transactions),min_support=0.6)
end = time.time()
print("Computational time for 2D dataset Fpgrowth : %f" % (end-start))


data20D =[[1,0,3,4,5,6,7,8,2,9,1,0,9,4,5,2,0,8,2,7],
          [0,2,5,3,5,6,9,8,2,9,1,0,8,1,0,3,1,7,1,6],
          [1,0,3,4,5,6,7,8,2,9,1,0,7,0,1,5,9,6,0,4],
          [1,0,3,4,5,6,7,8,2,9,1,0,6,3,9,6,8,5,9,2],
          [1,0,3,4,5,6,7,8,2,9,1,0,9,5,8,7,7,4,8,1],
          [1,0,3,4,5,6,7,8,2,9,1,0,8,7,6,8,5,3,6,3],
          [1,0,3,4,5,6,7,8,2,9,1,0,7,8,5,9,2,2,5,4],
          [1,0,3,4,5,6,7,8,2,9,1,0,5,9,3,1,4,1,3,1],
          [1,0,3,4,5,6,7,8,2,9,1,0,0,1,1,0,0,0,2,4],
          [1,0,3,4,5,6,7,8,2,9,1,0,1,2,0,2,1,3,1,5]]
my_transactionencoder= TransactionEncoder()
my_transactionencoder.fit(data20D)
encoded_transactions = my_transactionencoder.transform(data20D)

start = time.time()
frequent_itemsets=fpgrowth(pd.DataFrame(encoded_transactions),min_support=0.6)
end = time.time()
print("Computational time for 20D dataset Fpgrowth : %f" % (end-start))


