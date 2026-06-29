import torch
import torch.nn.functional as F
from kmeans_gpu import kmeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NormalizeFeaTorch(features):
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    return torch.mm(r_mat_inv, features)


def get_Similarity(fea_mat1, fea_mat2):
    return F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)


def clustering(feature, cluster_num):
    try:
        predict_labels, cluster_centers = kmeans(
            X=feature, num_clusters=cluster_num, distance="euclidean", device=device
        )
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        predict_labels, cluster_centers = kmeans(
            X=feature, num_clusters=cluster_num, distance="euclidean", device=torch.device("cpu")
        )
    return predict_labels.numpy(), cluster_centers


def euclidean_dist(x, y, root=False):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    if root:
        dist = dist.clamp(min=1e-12).sqrt()
    return dist
