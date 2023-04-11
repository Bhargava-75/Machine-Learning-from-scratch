import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    normalized_X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())*2-1
    normalized_X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())*2-1
    return normalized_X_train,normalized_X_test


def plot_metrics(metrics) -> None:
    # plot and save the results
    K=[]
    accuracy= []
    precision= []
    recall = []
    f1_score= []
    for i in range(len(metrics)):
        K.append("K = "+str(metrics[i][0]))
        accuracy.append(metrics[i][1])
        precision.append(metrics[i][2])
        recall.append(metrics[i][3])
        f1_score.append(metrics[i][4])
    barWidth = 0.2
    fig = plt.subplots(figsize =(12, 8))


    br1 = np.arange(len(K))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.bar(br1, accuracy, color ='#f58f71', width = barWidth,
            edgecolor ='grey', label ='Accuracy')
    plt.bar(br2, precision, color ='lightgreen', width = barWidth,
            edgecolor ='grey', label ='Precision')
    plt.bar(br3, recall, color ='skyblue', width = barWidth,
            edgecolor ='grey', label ='Recall')
    plt.bar(br4, f1_score, color ='yellow', width = barWidth,
            edgecolor ='grey', label ='F1_Score')

    plt.xlabel('Metrics', fontweight ='bold', fontsize = 15)
    plt.ylabel('Values', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(K))], K,fontsize ='10')
    plt.title("Performance Metrics",fontweight ='bold', fontsize = 20)
    
    plt.legend()
    plt.show()
    plt.savefig('Performance Metrics.png')