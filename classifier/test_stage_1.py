import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from train import MyDataset
from train import NET

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils import outlier


def metrics(true_label, predict_label, num_known, distance_scores=None):
    """
    Extended evaluation metrics for open-set classification.

    Parameters
    ----------
    true_label : np.ndarray
        Ground-truth labels (known classes = [0..num_known-1], unknown = -1).
    predict_label : np.ndarray
        Predicted labels (same convention, unknown = -1).
    num_known : int
        Number of known classes.
    distance_scores : np.ndarray, optional
        Confidence / distance scores per sample (lower = more confident).
        Used for AUROC.

    Returns
    -------
    results : dict
        Dictionary with all metrics.
    """

    results = {}

    tkr, tur, kp, fkr, accuracy = metrics_stage_1(true_label, predict_label)
    results["TKR"] = tkr
    results["TUR"] = tur
    results["KP"] = kp
    results["FKR"] = fkr
    results["ACC"] = accuracy

    # classification report - ignore unknowns for per-class metrics
    valid_idx = true_label != -1
    if np.any(valid_idx):
        report = classification_report(
            true_label[valid_idx],
            predict_label[valid_idx],
            labels=list(range(num_known)),
            zero_division=0,
            output_dict=True
        )
        results["classification_report"] = report

    # confusion matrix including unknowns
    labels_with_unknown = list(range(num_known)) + [-1]
    cm = confusion_matrix(true_label, predict_label, labels=labels_with_unknown)
    results["confusion_matrix"] = cm

    # AUROC for known vs unknown
    if distance_scores is not None:
        y_true = (true_label == -1).astype(int) # 1 = unknown, 0 = known
        auroc = roc_auc_score(y_true, distance_scores)
        results["AUROC"] = auroc

    return results

def metrics_stage_1(true_label, predict_label):
    num_samples = predict_label.shape[0]
    ones = np.ones(num_samples)
    # TKR:
    #  A: the number of known samples are accepted / B: the number of known samples
    a = np.sum(np.logical_and(predict_label != (-ones), true_label != (-ones)))
    b = np.sum(true_label != (-ones))
    tkr = a / b

    # TUR:
    #  A: the number of unknown are rejected / B: the number of unknown samples
    a = np.sum(np.logical_and(true_label == (-ones), predict_label == (-ones)))
    b = np.sum(true_label == (-ones))
    tur = a / b

    # KP:
    # A: the number of known samples are accurately classified / the number of all accepted samples
    a = np.sum(true_label[true_label != (-ones)] == predict_label[true_label != (-ones)])
    b = np.sum(predict_label != (-ones))
    if (b == 0):
        kp = 0
    else:
        kp = a / b

    # FKR:
    # A: the number of unknown samples are accurately rejected / the number of all rejected samples
    a = np.sum(true_label[true_label == (-ones)] == predict_label[true_label == (-ones)])
    b = np.sum(predict_label == (-ones))
    fkr = a / b

    # Known Accuracy
    accuracy = []
    for ii in range(num_known):
        a = np.sum(np.logical_and(true_label == (ii * ones), predict_label == (ii * ones)))
        b = np.sum(true_label == (ii * ones))
        if b != 0:
            acc = a / b
            accuracy.append(acc)
        else:
            accuracy.append(-1)
    return tkr, tur, kp, fkr, accuracy


if __name__ == "__main__":
    tips = '(xyz_for_loss_curve)'
    my_index = 1
    semantic_dim = 128
    expand_semantic_dim = 128
    time_size = 1
    gamma = 0.75
    margin = 8
    num_known = 18
    num_unknown = 6
    num_total = num_known + num_unknown
    model_path = "./model/S3R/group_" + str(my_index) + \
                 "_margin_" + str(margin) + \
                 "_dim_" + str(semantic_dim) + \
                 "_length_" + str(time_size) + \
                 "_gamma_" + str(gamma) + \
                 "_tips_" + tips + \
                 ".pkl"
    model_paras = torch.load(model_path)
    # load data
    known_train_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-known_for_train', len_time=time_size, gamma=gamma, size=512)
    known_test_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-known_for_test', len_time=time_size, gamma=gamma, size=512)
    unknown_test_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-unknown', len_time=time_size, gamma=gamma,size=512)
    known_train_loader = DataLoader(dataset=known_train_data, batch_size=1, shuffle=True, drop_last=False)
    known_test_loader = DataLoader(dataset=known_test_data, batch_size=1, shuffle=True, drop_last=False)
    unknown_test_loader = DataLoader(dataset=unknown_test_data, batch_size=1, shuffle=True, drop_last=False)
    # configure network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = NET(in_channels=1, input_size=[int(512*time_size), 512], semantic_dim=semantic_dim, num_class=num_known, device=device).to(device)
    Net.load_state_dict(model_paras)
    # get centers, thresholds from train data
    with torch.no_grad():
        Net.eval()
        # read all semantics of training samples
        train_X = torch.zeros((len(known_train_data), semantic_dim))
        train_Y = torch.zeros(len(known_train_data))
        for i, data in enumerate(tqdm(known_train_loader)):
            x_batch, y_batch, z_batch, label = data
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            _, train_X[i], _, _, _, _ = Net(x_batch, y_batch, z_batch)
            train_Y[i] = label

  
        # ---------- REFACTORED ----------
        # compute class stats with regularized cov + precision
        # train_X = torch tensor (N, D)
        # train_Y = torch tensor (N,)
        class_means, precisions, covariances = outlier.compute_class_stats(train_X, train_Y, num_known, semantic_dim, reg_lambda=1e-3, use_ledoit=False)
        # compute per-class percentile thresholds (distance domain)
        percentile = 95.0
        thresholds = outlier.per_class_thresholds_percentile(train_X, train_Y, class_means, precisions, percentile)
        # thresholds is array shape (num_known, ) - distances in same units as sqrt(d2)

        # fit EVT weibull models to each class tail
        use_evt = True
        weibull_models = None
        if use_evt:
            weibull_models =  outlier.fit_weibull_per_class(train_X, train_Y, class_means, precisions, tail_size=20)
        # ---------- REFACTORED ----------

        # read all testing data
        test_X = torch.zeros((len(known_test_data) + len(unknown_test_data), semantic_dim))
        test_X_expand = torch.zeros((len(known_test_data) + len(unknown_test_data), expand_semantic_dim * 3))
        test_Y = torch.zeros(len(known_test_data) + len(unknown_test_data))
        # confusion_matrix = np.zeros((num_known, num_known))
        for i, data in enumerate(tqdm(known_test_loader)):
            x_batch, y_batch, z_batch, label = data
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            predict, test_X[i], _, _, _, test_X_expand[i] = Net(x_batch, y_batch, z_batch)
            test_Y[i] = label
            # pre = torch.max(predict.data, 1)[1]
            # confusion_matrix[int(label.cpu().numpy())][int(pre.cpu().numpy())] += 1
        for i, data in enumerate(tqdm(unknown_test_loader)):
            x_batch, y_batch, z_batch, label = data
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            _, test_X[len(known_test_data) + i], _, _, _, test_X_expand[len(known_test_data) + i] = Net(x_batch, y_batch, z_batch)
            test_Y[len(known_test_data) + i] = label

        # ---------- REFACTORED ----------
        # start evaluation at stage 1
        # test_X: torch tensor(M, D) OR numpy array (M, D)
        if hasattr(test_X, "cpu"):
            test_X_np = test_X.cpu().numpy()
        else:
            test_X_np = np.array(test_X)

        # get squared Mahalanobis distances (M, C)
        d2 = outlier.batch_mahalanobis_sq(test_X_np, class_means, precisions) # squared distances
        d = np.sqrt(np.maximum(d2, 0.0)) # normal distances

        # decision by threshold: compute d - threshold (broadcast)
        x_ct = d - thresholds[np.newaxis, :] # shape (M,C)

        # compute EVT outlier probabilities
        if use_evt and outlier.SCIPY_AVAILABLE and weibull_models is not None:
            weibull_probs = outlier.weibull_outlier_probability(d, weibull_models) # (M, C)
        else:
            weibull_probs = None
        
        
        label_hat = np.zeros(test_X_np.shape[0], dtype=np.int32)
        tur_mistake = np.zeros((num_total - num_known, num_known))
        for i in range(test_X_np.shape[0]):
            # first criterion: mahalanobis distance thresholding
            if np.min(x_ct[i]) > 0:
                label_hat[i] = -1 # unknown
            else:
                cand = int(np.argmin(x_ct[i])) # nearest known class

                # EVT refinement
                if weibull_probs is not None:
                    # tail probability for being an outlier
                    evt_cutoff = 0.5 # TODO tune on validation
                    if weibull_probs[i, cand] > evt_cutoff:
                        label_hat[i] = -1
                    else:
                        label_hat[i] = cand
                else:
                    label_hat[i] = cand

            # bookkeeping: if this was actually an unknown sample but we missclassified it
            if test_Y[i] >= num_known and label_hat[i] != -1:
                tur_mistake[int(test_Y[i] - num_known), cand] += 1

        # normalize ground-truth labels: unknown -> -1
        test_Y_normalized = test_Y.cpu().numpy().copy()
        for i in range(test_Y_normalized.shape[0]):
            if test_Y_normalized[i] >= num_known:
                test_Y_normalized[i] = -1
        # ---------- REFACTORED ----------

        # tkr, tur, kp, fkr, accuracy = metrics_stage_1(test_Y_normalized, label_hat)
        
        results = metrics(test_Y_normalized,
                                  label_hat,
                                  num_known=num_known,
                                  distance_scores=np.min(d, axis=1),
                                  )
                
        tkr = results["TKR"]
        tur = results["TUR"]
        kp = results["KP"]
        fkr = results["FKR"]
        accuracy = results["ACC"]
        class_report = results["classification_report"]
        cm = results["confusion_matrix"]
        auroc = results["AUROC"]
        
        supervised_acc = []
        for xi in range(num_known):
            supervised_acc.append(cm[xi][xi] / np.sum(cm[xi]))
        print("supervised accuracy:\n", supervised_acc)
        print("known accuracy:", np.round(accuracy, 4))
        print("mean known accuracy:", np.mean(np.round(accuracy, 4)))
        # print("tkr, tur, kp, fkr:", tkr, tur, kp, fkr)
        print("tkr: {} | tur: {} | kp: {} | AUROC: {}".format(tkr, tur, kp, auroc))
        print("tur mistake:\n", tur_mistake)

        np.save('./test_X.npy', test_X.cpu().numpy())
        np.save('./test_Y.npy', test_Y.cpu().numpy())
        np.save('./label_hat.npy', label_hat)
        np.save('./thresholds.npy', thresholds.cpu().numpy()) # per class thresholds
        np.save('./centers.npy', class_means.cpu().numpy()) # class centers
        np.save('./dist_matrix.npy', d) # distances (test x classes)
        np.save('./test_X_expand.npy', test_X_expand.cpu().numpy())

        # record semantics
        np.save('./semantic/S3R/test_X.npy', test_X.cpu().numpy())
        np.save('./semantic/S3R/test_Y.npy', test_Y.cpu().numpy())
        np.save('./semantic/S3R/test_X_expand.npy', test_X_expand.cpu().numpy())