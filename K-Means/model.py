import numpy as np
from tqdm import tqdm
class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon

    def find_error(self,X,Y):
        np_X = np.array(X)
        np_Y = np.array(Y)
        return np.linalg.norm(np_X-np_Y).mean()
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        
        self.cluster_centers=[]
        cluster_centers_copy=[]
        num_points = X.shape[0]
        num_dim = X.shape[1]
        near_cluster_center = np.zeros(shape=(num_points,self.num_clusters))
        while True:
            rand = np.random.randint(0,num_points)
            if (X[rand].tolist() not in self.cluster_centers):
                self.cluster_centers.append(X[rand].tolist())
            if(len(self.cluster_centers)==self.num_clusters):
                break
        for _ in tqdm(range(max_iter)):
            # Assign each sample to the closest prototype
            cluster_centers_copy = self.cluster_centers.copy()
            near_cluster_center = np.zeros(shape=(num_points,self.num_clusters))
            for i in range(num_points):
                dist = np.array([np.linalg.norm(X[i]-j) for j in self.cluster_centers])
                min_dist=np.argmin(dist)
                near_cluster_center[i,min_dist] = 1
            count_vals_each_cluster = np.sum(near_cluster_center, axis = 0)
            tem = np.matmul(near_cluster_center.T,X)
            for k in range(self.num_clusters):
                self.cluster_centers[k] = tem[k]/count_vals_each_cluster[k]
            #Finding error
            error = self.find_error(self.cluster_centers,cluster_centers_copy)
            if(error<=self.epsilon):
                break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        count =0
        predicted_clusters = np.array([0]*X.shape[0])
        for i in X:
            dist = np.array([np.linalg.norm(i-j) for j in self.cluster_centers])
            min_dist=np.argmin(dist)
            predicted_clusters[count] = min_dist
            count=count+1
        return predicted_clusters 

    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        predicted_clusters = self.predict(X)
        num_points=X.shape[0]
        new_X = X.copy()
        for i in range(num_points):
            new_X[i] = self.cluster_centers[predicted_clusters[i]]
        return new_X
