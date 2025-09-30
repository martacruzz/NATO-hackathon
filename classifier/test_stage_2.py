"""
Stage 2: Unknown Clustering
---------------------------

Design choice:
- Stage 1 (known vs unknown detection) already uses EVT (Weibull tail fitting)
  for robust open-set recognition.
- Stage 2's purpose is different: estimate how many *unknown* classes (u) exist
  among the rejected samples.
- Since samples here are *already rejected* by EVT, we don't re-apply EVT.
  Instead, we use Mahalanobis distances + a simple percentile heuristic to
  check if all unknowns form a single cluster (u=1) or if clustering is needed.

Summary:
- If unknowns look compact (u=1), report accuracy + UP.
- Otherwise, run KMeans (2-14) and pick the best cluster count with DB index.
"""

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics

from utils.outlier import batch_mahalanobis_sq


if __name__ == "__main__":
    num_known = 18
    num_unknown = 6

    # load data from stage1
    # embeddings
    test_X = np.load('./test_X.npy')
    test_X_expand = np.load('./test_X_expand.npy')
    test_X = np.hstack((test_X, test_X_expand))
    test_Y = np.load('./test_Y.npy')
    
    label_hat = np.load('./label_hat.npy') # predictions
    theta = np.load('./thresholds.npy') # per class thresholds (stage 1)
    class_centers = np.load('./centers.npy') # known class means
    dist_matrix = np.load('./dist_matrix.npy') # distances (test x classes)

    # normalize labels - known: [0...num_known]; unknown: -1
    test_Y_normalized = test_Y.copy()
    for xi in range(test_Y_normalized.shape[0]):
        if test_Y_normalized[xi] >= num_known:
            test_Y_normalized[xi] = -1

    # filter predicted unknown samples
    predict_unknown_X = torch.FloatTensor(test_X[label_hat == -1])
    predict_unknown_Y = torch.LongTensor(test_Y[label_hat == -1])
    ones = np.ones(test_Y.shape[0])
    
    # estimate number of unknown classes (u)
    samples = predict_unknown_X.cpu().numpy()
    
    # compactness test (u = 1 ?)
    covariance_mat = np.cov(samples, rowvar=False, bias=True)
    precisions = np.linalg.pinv(covariance_mat)
    centers = torch.mean(predict_unknown_X, dim=0)

    d2 = batch_mahalanobis_sq(samples, centers, precisions[np.newaxis, :, :])
    d = np.sqrt(np.maximum(d2.flatten()), 0.0)

    # percentile cutoff - EVT already handled in Stage 1
    theta_u1 = np.percentile(d, 95.0) # TODO change this as needed
    if theta_u1 <= np.max(theta):
        print("u = 1")
        u = 1

        # calculate unknown accuracy
        a = np.sum(np.logical_and(test_Y_normalized == (-ones), label_hat == (-ones)))
        b = np.sum(test_Y_normalized == (-ones))
        unknown_acc = a / b
        print("unknown_acc:", unknown_acc)

        # calculate UP (unknown precision)
        c = predict_unknown_X.shape[0]
        UP = a / c
        print("UP:", UP)
    else:
        # clustering (if not compact)
        scalar = MinMaxScaler()
        predict_unknown_X = scalar.fit_transform(predict_unknown_X)

        DB = []
        SC = []
        for ui in range(2, 15):
            Cluster = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(predict_unknown_X)
            pre_label = Cluster.labels_
            sc = metrics.silhouette_score(predict_unknown_X, pre_label)
            db = metrics.davies_bouldin_score(predict_unknown_X, pre_label)
            print(ui, sc, db)
            DB.append(db)
            SC.append(sc)
        
        DB = np.array(DB)
        SC = np.array(SC)
        u = np.argmin(DB) + 2
        print("u=", u)
        
        Cluster = KMeans(n_clusters=u, init='k-means++', random_state=51).fit(predict_unknown_X)
        pred_label = Cluster.labels_
        
        # build confusion matrix against ground truth
        confusion_mat = np.zeros((u, num_known+num_unknown))
        for xi in range(predict_unknown_X.shape[0]):
            confusion_mat[int(pred_label[xi])][int(predict_unknown_Y[xi])] += 1
        confusion_mat = confusion_mat[:, num_known:]
        print("confusion_mat:", confusion_mat)
        
        # Dominant cluster per unknown class
        dominate_sample = np.zeros(num_unknown)
        for row in range(u):
            for col in range(num_unknown):
                if confusion_mat[row][col] >= np.sum(confusion_mat[row]) * 0.5 and np.argmax(confusion_mat[:, col]) == row:
                    dominate_sample[col] = confusion_mat[row][col]
        print("dominate_sample:", dominate_sample)
        
        # calculate the unknown accuracy
        unknown_acc = []
        for clas in range(num_unknown):
            a = np.sum(test_Y == int(clas+num_known) * ones)
            unknown_acc.append(dominate_sample[clas] / a)
        print("unknown_acc:", unknown_acc)
        print("mean unknown_acc:", np.mean(unknown_acc))
        
        # unknown precision (UP)
        up = np.sum(dominate_sample) / predict_unknown_X.shape[0]
        print("up:", up)