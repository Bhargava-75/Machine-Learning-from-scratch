import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        cov_matrix = np.cov(X,rowvar=False)
        eignvals, eignvecs = np.linalg.eigh(cov_matrix)
        eignvecs = eignvecs[:, ::-1]
        self.components = eignvecs[:, :self.n_components]
        
    def transform(self, X) -> np.ndarray:
        # transform the data
        transformed_X = np.dot(X, self.components)
        return transformed_X

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w = np.zeros(X.shape[1])
        self.b= 0

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            rand = np.random.randint(X.shape[0])
            xi,yi = X[rand],y[rand]
            #Calculating the Loss
            cond = yi*(np.dot(xi,self.w)+self.b)>=1
            dw = self.w if cond else C*self.w - yi*xi
            db = 0 if cond else -C*yi
            
            #Update the parameters
            self.w -= learning_rate*dw
            self.b -= learning_rate*db
    
    def predict(self, X) -> np.ndarray:
        #make predictions for the given data
        predictions = np.dot(X,self.w) +self.b
        return predictions

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)
        


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        self.precision=[]
        self.recall=[]
        self.f1_score=[]
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, learning_rate,num_iters,C) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.models = [SupportVectorModel() for _ in range(self.num_classes)]
        
        for i in range(self.num_classes):
            y_binary = np.where(y == self.classes[i], 1, -1)
            self.models[i].fit(X, y_binary, learning_rate, num_iters,C)
        self.metrices(X,y)

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            scores[:, i] = self.models[i].predict(X)
        y_pred = np.argmax(scores, axis=1)
        return self.classes[y_pred]
    
    def metrices(self,X,y):
        y_pred = self.predict(X)
        for i in range(10):
            tp_i = sum((y== i) & (y_pred == i))
            fp_i = sum((y!= i) & (y_pred == i))
            fn_i = sum((y== i) & (y_pred != i))
            self.precision.append(tp_i/(tp_i +fp_i))
            self.recall.append(tp_i/(tp_i+fn_i))
            self.f1_score.append(2*self.precision[i]*self.recall[i]/(self.precision[i]+self.recall[i]))
            
    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        precision = sum(self.precision)/self.num_classes
        return round(precision,4)
    
    def recall_score(self, X, y) -> float:
        recall = sum(self.recall)/self.num_classes
        return round(recall,4)
    
    def f1_scores(self, X, y) -> float:
        f1_score = sum(self.f1_score)/self.num_classes
        return round(f1_score,4)
