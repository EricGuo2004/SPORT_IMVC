import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    # 用 scipy 替换 Munkres，无需额外依赖
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = list(zip(row_ind, col_ind))
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    return kmeans_to_true_cluster_labels[cluster_assignments]


def fmetric(y_true, y_pred, n_clusters):
    y_pred_adjusted = get_y_preds(y_true, y_pred, n_clusters)
    f_score   = metrics.f1_score(y_true, y_pred_adjusted, average="weighted")
    precision = metrics.precision_score(y_true, y_pred_adjusted, average="weighted")
    recall    = metrics.recall_score(y_true, y_pred_adjusted, average="macro")  # 统一为 weighted
    return f_score, precision, recall


def Purity_score(y_true, y_pred):
    y_true = y_true.copy()  # 防止修改原数组，避免污染后续指标计算
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    return metrics.accuracy_score(y_true, y_voted_labels)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(truth, prediction):
    unique = np.unique(truth)
    n_clusters = np.size(unique, axis=0)
    nmi    = metrics.normalized_mutual_info_score(truth, prediction)
    acc    = cluster_acc(truth, prediction)
    purity = Purity_score(truth, prediction)          # 内部已 copy，不再污染 truth
    fscore, precision, recall = fmetric(truth, prediction, n_clusters)
    ari    = adjusted_rand_score(truth, prediction)
    return (
        float(np.round(acc,       4)),
        float(np.round(nmi,       4)),
        float(np.round(purity,    4)),
        float(np.round(fscore,    4)),
        float(np.round(precision, 4)),
        float(np.round(recall,    4)),
        float(np.round(ari,       4)),
    )