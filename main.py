import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans

import load_data as loader
from Nmetrics import evaluate
from network import Network
from datasets import Data_Sampler, TrainDataset_Com, TrainDataset_All
from utils import get_Similarity, clustering


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def collect_features(model, args, device, X, Y, batch_size):
    feature_list = [[] for _ in range(args.V)]
    dataset = TrainDataset_Com(X, Y)
    sampler = Data_Sampler(dataset, shuffle=False, batch_size=batch_size, drop_last=False)
    loader_all = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler)
    with torch.no_grad():
        for xs, _ in loader_all:
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            zs, _ = model(xs)
            for v in range(args.V):
                feature_list[v].extend(zs[v].cpu().tolist())
    for v in range(args.V):
        feature_list[v] = torch.tensor(feature_list[v], dtype=torch.float32)
    return feature_list


def mean_cosine_similarity_shared_views(fea_all, missindex, i, j, num_views):
    """Mean cosine similarity between instances i and j on views both observe."""
    acc = 0.0
    cnt = 0
    for v in range(num_views):
        if missindex[i, v] == 1 and missindex[j, v] == 1:
            a = fea_all[v][i : i + 1]
            b = fea_all[v][j : j + 1]
            acc += F.cosine_similarity(a, b).item()
            cnt += 1
    if cnt == 0:
        return float("-inf")
    return acc / cnt


def print_training_args(args, stage_name):
    if getattr(args, "quiet", False):
        return
    parts = []
    for key in sorted(vars(args).keys()):
        value = getattr(args, key)
        parts.append(f"{key}={value}")
    print(f"[{stage_name} Params] " + " | ".join(parts))




def compute_soft_labels(z, miss_vec, V, tau_w):
    """
    Compute w_ij per Eq.5.
    z        : list of V tensors, each (B, d)
    miss_vec : list of V tensors, each (B,), 1=observed 0=missing
    Returns  : (B, B) soft label matrix, stop-gradiented
    """
    B      = z[0].shape[0]
    device = z[0].device

    w     = torch.zeros(B, B, device=device)
    count = torch.zeros(B, B, device=device)

    for v in range(V):
        obs_v    = (miss_vec[v].to(device) == 1).float()             # (B,)
        both_obs = obs_v.unsqueeze(1) * obs_v.unsqueeze(0) # (B, B)

        # Safe L2-normalize: clamp avoids NaN for zero (missing) embeddings
        norms    = z[v].norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_v_norm = z[v] / norms                            # (B, d)

        S_vv   = torch.mm(z_v_norm, z_v_norm.t())         # (B, B), cosine similarities
        # Numerically stable Eq.5 form:
        # exp(S/tau_w) / exp(1/tau_w) == exp((S-1)/tau_w), exponent <= 0
        contrib = torch.exp((S_vv - 1.0) / tau_w)  # (B, B)

        w     = w     + both_obs * contrib
        count = count + both_obs

    # Where count > 0: average; elsewhere: 0  (|V_i ∩ V_j| = 0 case)
    w_ij = torch.where(count > 0, w / count, torch.zeros_like(w))
    return w_ij.detach()   # stop-gradient: supervision signal only




def compute_contrastive_loss(z, miss_vec, w_ij, V, tau):
    """
    Compute L_SL per Eq.7, summing over all ordered pairs (v, u), v != u.

    For each ordered pair (v, u):
      anchor i  : samples observed in view v  (i = 1..Nv)
                  with indicator 1[i in N_{v,u}]  (also observed in u)
      column j  : samples observed in view u  (j = 1..Nu)
      numerator : sum_j  w_ij * exp(S(hi^v, hj^u) / tau)
      denominator: sum_j exp(S(hi^v, hj^v)/tau) + sum_j exp(S(hi^v, hj^u)/tau)
      loss_{v,u}: -1/Nv * sum_i  1[i in N_{v,u}] * log(num / denom)
    """
    device = z[0].device
    total_loss = torch.tensor(0., device=device)

    # Pre-normalize all features once
    z_norm = []
    for v in range(V):
        norms = z[v].norm(dim=1, keepdim=True).clamp(min=1e-8)
        z_norm.append(z[v] / norms)

    for v in range(V):
        obs_v     = miss_vec[v] == 1          # (B,) bool
        obs_v_idx = torch.where(obs_v)[0]     # (Nv,) indices
        Nv        = len(obs_v_idx)
        if Nv == 0:
            continue

        hv = z_norm[v][obs_v_idx]             # (Nv, d)

        for u in range(V):
            if u == v:
                continue

            obs_u     = miss_vec[u] == 1      # (B,) bool
            obs_u_idx = torch.where(obs_u)[0] # (Nu,) indices
            Nu        = len(obs_u_idx)
            if Nu == 0:
                continue

            hu = z_norm[u][obs_u_idx]         # (Nu, d)

          
            indicator = obs_u[obs_v_idx].float()   # (Nv,)
            if indicator.sum() < 2:
                continue

            # w submatrix rows=obs in v, cols=obs in u  ->  (Nv, Nu)
            w_sub = w_ij[obs_v_idx][:, obs_u_idx]  # (Nv, Nu)

            S_vu = torch.mm(hv, hu.t()) / tau      # (Nv, Nu)
            S_vv = torch.mm(hv, hv.t()) / tau      # (Nv, Nv)

            # numerator: sum_j  w_ij * exp(S_vu)   -> (Nv,)
            numerator = torch.sum(w_sub * torch.exp(S_vu), dim=1)

            # denominator: intra-view + cross-view   -> (Nv,)
            denom = torch.exp(S_vv).sum(dim=1) + torch.exp(S_vu).sum(dim=1)

            log_probs = torch.log(numerator / (denom + 1e-12) + 1e-12)   # (Nv,)

            # -1/Nv * sum_i  indicator_i * log_probs_i
            loss_vu    = -torch.sum(indicator * log_probs) / Nv
            total_loss = total_loss + loss_vu

    return total_loss




def get_decomp_losses(shared_c, private_s, tau=0.75, t_pow=3):
    """
    shared_c  : list of V tensors, each (K, dc)
    private_s : list of V tensors, each (K, d)
    """
    V      = len(shared_c)
    device = shared_c[0].device
    K      = shared_c[0].shape[0]

    L_shared  = torch.tensor(0., device=device)
    L_private = torch.tensor(0., device=device)

    
    for v in range(V):
        cv_norm = F.normalize(shared_c[v],  dim=1)   # (K, dc)
        sv_norm = F.normalize(private_s[v], dim=1)   # (K, d)

        for u in range(V):
            if v == u:
                continue

            cu_norm = F.normalize(shared_c[u],  dim=1)   # (K, dc)
            su_norm = F.normalize(private_s[u], dim=1)   # (K, d)

          
            sim_cv_cu = torch.mm(cv_norm, cu_norm.t()) / tau   # (K, K)
            Q_vu      = F.softmax(sim_cv_cu, dim=1)            # (K, K)
            Q_diag    = torch.diag(Q_vu)                       # (K,)

            # Fix: sum over n=1..t_pow,  (1/K) sum_k (1 - Q_k^vu)^n / n
            loss_shared_vu = torch.tensor(0., device=device)
            for n in range(1, int(t_pow) + 1):
                loss_shared_vu = loss_shared_vu + torch.sum(
                    torch.pow(1.0 - Q_diag, n) / n
                ) / K
            L_shared = L_shared + loss_shared_vu

           
            align      = torch.mm(sv_norm, su_norm.t())   # (K, K)
            align_diag = torch.diag(align)                # (K,) — same-k pairs only
            L_private  = L_private + torch.abs(align_diag).sum() / K

    return L_shared, L_private




class ExactCayleyPrototypeLearner(nn.Module):
    """
    Prototype matrix mu (V, K, d) initialised via K-means (Eq.8).
    Shared subspace basis U from Cayley transform (Eq.10): U = (I+A)^{-1}(I-A),
    A = W - W^T, first d_c columns.
    Decomposition (Eq.9): c_v = mu_v U U^T, s_v = mu_v (I - U U^T).
    """
    def __init__(self, views, feature_dim, num_clusters, dc_dim, pretrained_features):
        super(ExactCayleyPrototypeLearner, self).__init__()
        self.views = views
        self.d  = feature_dim
        self.K  = num_clusters
        self.dc = dc_dim
        if dc_dim > feature_dim:
            raise ValueError("d_c must be <= feature_dim.")

        self.W = nn.Parameter(torch.randn(self.d, self.d) * 0.01)

        mu_init_list = []
        for v in range(views):
            _, centers = clustering(pretrained_features[v], self.K)
            mu_init_list.append(centers.clone().to(pretrained_features[v].device))
        self.mu_param = nn.Parameter(torch.stack(mu_init_list, dim=0))

    def _get_u_basis(self):
        A   = self.W - self.W.t()
        eye = torch.eye(self.d, device=A.device, dtype=A.dtype)
        U_full = torch.linalg.solve(eye + A, eye - A)
        return U_full[:, : self.dc]

    def forward(self):
        U             = self._get_u_basis()
        UUT           = torch.mm(U, U.t())
        I_minus_UUT   = torch.eye(self.d, device=U.device, dtype=U.dtype) - UUT

        all_mu, all_c, all_s = [], [], []
        for v in range(self.views):
            mu_v = self.mu_param[v]
            c_v  = torch.mm(mu_v, UUT)
            s_v  = torch.mm(mu_v, I_minus_UUT)
            all_mu.append(mu_v)
            all_c.append(c_v)
            all_s.append(s_v)

        return all_mu, all_c, all_s


def pretrain(model, opt_pre, args, device, X_com, Y_com, X, Y):
    train_dataset = TrainDataset_Com(X_com, Y_com)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    _quiet = getattr(args, "quiet", False)
    t_progress = tqdm(
        range(args.pretrain_epochs),
        desc="Pretraining",
        dynamic_ncols=True,
        disable=_quiet,
    )

    for epoch in t_progress:
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, ys) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)

            opt_pre.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
            loss = sum(loss_list)
            loss.backward()
            opt_pre.step()

    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    all_dataset = TrainDataset_Com(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                             batch_sampler=batch_sampler_all)
    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()

    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])

    return fea_emb




def train_align(model, proto_learner, opt_align, args, device, X, Y, Miss_vecs, missindex):
    train_dataset = TrainDataset_All(X, Y, Miss_vecs)
    sampler       = Data_Sampler(train_dataset, shuffle=True,
                                 batch_size=args.Batch_Rob, drop_last=True)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_sampler=sampler)
    recon_loss_fn = torch.nn.MSELoss().to(device)

    alpha   = args.para_loss[0]  
    beta    = args.para_loss[1]   
    eta     = args.para_loss[2]  
    lambda_ = args.para_loss[3]  
    gamma   = args.para_loss[4]   
    w_neigh = args.para_loss[5]   

    for _ in tqdm(
        range(args.FineTuning_epochs),
        desc="Fine-Tuning",
        disable=getattr(args, "quiet", False),
    ):
        for _, (x, _, miss_vec) in enumerate(train_loader):
            opt_align.zero_grad()

            for v in range(args.V):
                x[v]        = torch.squeeze(x[v]).to(device)
                miss_vec[v] = torch.squeeze(miss_vec[v]).to(device)

            z, xr = model(x)
            mu_prototypes, shared_c, private_s = proto_learner()

            # Reconstruction loss 
            rec_losses = []
            for v in range(args.V):
                vis = miss_vec[v] == 1
                if vis.sum() > 0:
                    rec_losses.append(recon_loss_fn(x[v][vis], xr[v][vis]))
            Loss_Rec = sum(rec_losses) if rec_losses else torch.tensor(0., device=device)

            # Soft labels
            w_ij = compute_soft_labels(z, miss_vec, args.V, args.tau_w)

            # Contrastive loss 
            Loss_SL = compute_contrastive_loss(
                z, miss_vec, w_ij, args.V, args.soft_temp
            )

            # Decomposition losses 
            l_share, l_priv = get_decomp_losses(
                shared_c, private_s, tau=args.soft_temp, t_pow=args.t_pow
            )

            # L_decomp (
            L_decomp   = lambda_ * l_share + eta * l_priv
            Total_Loss = Loss_Rec + alpha * Loss_SL + beta * L_decomp

            Total_Loss.backward()
            opt_align.step()

    fea_all = collect_features(model, args, device, X, Y, args.batch_size)
    with torch.no_grad():
        all_mu_prototypes, _, _ = proto_learner()
        all_mu_prototypes = [p.detach().cpu() for p in all_mu_prototypes]

    fea_final = [fea_all[v].clone() for v in range(args.V)]
    total_num = fea_final[0].shape[0]
    for i in tqdm(
        range(total_num),
        desc="Hybrid Imputation",
        dynamic_ncols=True,
        disable=getattr(args, "quiet", False),
    ):
        observed_views = [v for v in range(args.V) if missindex[i, v] == 1]
        for miss_v in range(args.V):
            if missindex[i, miss_v] == 0:
                if len(observed_views) == 0:
                    continue

                tgt_proto = all_mu_prototypes[miss_v]
                if tgt_proto.shape[0] == 0:
                    continue

                common_k = tgt_proto.shape[0]
                for obs_v in observed_views:
                    common_k = min(common_k, all_mu_prototypes[obs_v].shape[0])
                if common_k <= 0:
                    continue

               
                mean_sim_k = torch.zeros(common_k)
                for obs_v in observed_views:
                    obs_proto = all_mu_prototypes[obs_v][:common_k]
                    sim_vec = get_Similarity(
                        obs_proto, fea_all[obs_v][i].unsqueeze(0)
                    ).squeeze(1)
                    mean_sim_k = mean_sim_k + sim_vec
                mean_sim_k = mean_sim_k / len(observed_views)
                best_idx = torch.argmax(mean_sim_k).item()
                proto_vec = tgt_proto[:common_k][best_idx]

                best_j = -1
                best_cross = float("-inf")
                for j in range(total_num):
                    if j == i or missindex[j, miss_v] != 1:
                        continue
                    s_ij = mean_cosine_similarity_shared_views(
                        fea_all, missindex, i, j, args.V
                    )
                    if s_ij > best_cross:
                        best_cross = s_ij
                        best_j = j

                if best_j >= 0 and best_cross > float("-inf"):
                    neigh_vec = fea_all[miss_v][best_j]
                    fea_final[miss_v][i] = gamma * proto_vec + (1 - gamma) * neigh_vec
                else:
                    fea_final[miss_v][i] = proto_vec

    return fea_final


# =============================================================================
# Argument parser & main
# =============================================================================

i_d = {
    0: "Caltech101_7",
    1: "LandUse_21",
    2: "Scene_15",
    3: "ALOI_100",
    4: "YouTubeFace10_4Views",
    5: "HandWritten",
    6: "EMNIST_digits_4Views",
    7: "AWA",
    8: "Leaves100",

    9: "News20",
    10: "Sources3",
    11: "ACM3025",
    12: "Animal",
    13: "BBC",
    14: "BBCSport",
    15: "BBCSport_2View1",
    16: "Caltech101_7_AllViews",
    17: "COIL100",
    18: "COIL20",
    19: "CUB_GoogLeNet_Doc2Vec_C10",
    20: "Digit4k",
    21: "ESP_Game",
    22: "Flower17",
    23: "Food_101",
    24: "HDigit",
    25: "Mfeat",
    26: "MNIST_3Views",
    27: "MSRC_v6",
    28: "MSRC",
    29: "MSRC_v1",
    30: "NUS_WIDE",
    31: "ORL",
    32: "PIE_face_10",
    33: "Prokaryotic",
    34: "Reuters_21578",
    35: "ThreeRing",
    36: "TwoMoon",
    37: "UCI_Digit6",
    38: "USPS",
    39: "VGGFace2_50",
    40: "WebKB",
    41: "WikipediaArticles",
    42: "Yale",
    43: "YaleB",
    44: "Youtube",
    45: "Caltech101_20",
    46: "CiteSeer",
    47: "Reuters",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser     = argparse.ArgumentParser(description="IMVC-CorrectedFormula")
parser.add_argument("--i_d",      type=int,   default =34)
parser.add_argument("--missrate", type=float, default=0.5)
parser.add_argument("--seed",     type=int,   default=12) 
parser.add_argument(
    "--quiet",
    action="store_true",
    help="Fewer logs; final block prints only the six metrics (ACC,NMI,F1,ARI,Recall,Precision).",
)

args_temp    = parser.parse_known_args()[0]
dataset_name = i_d[args_temp.i_d]
data_para    = loader.ALL_data[dataset_name]

optimal_para_loss = [1.6e-05, 5, 5, 10, 0.3, 0.7] # [alpha, beta, eta, lambda, gamma, w_neigh]
parser.add_argument("--dataset",         default=data_para)
parser.add_argument("--batch_size",      default=512,               type=int)
parser.add_argument("--Batch_Rob",       default=512,               type=int)
parser.add_argument("--lr_pre",          default=0.0005,            type=float)
parser.add_argument("--lr_finetuning",   default=0.00001,           type=float)
parser.add_argument("--para_loss",       default=optimal_para_loss, type=float, nargs="+",
                    help="[alpha, beta, eta, lambda, gamma, w_neigh] (6 values)")
parser.add_argument("--soft_temp",       default=0.75,               type=float,
                    help="tau: temperature for contrastive loss")
parser.add_argument("--tau_w",           default=0.5,               type=float,
                    help="tau_w: temperature for soft label construction")
parser.add_argument("--t_pow",           default=3,                 type=int,
                    help="t: polynomial degree in L_shared ")
parser.add_argument("--d_c",             default =128,              type=int,
                    help="dc: shared subspace dimension")
parser.add_argument("--pretrain_epochs", default=200,               type=int)
parser.add_argument("--FineTuning_epochs",default=100,               type=int)
parser.add_argument("--feature_dim",     default=256,               type=int)
parser.add_argument("--V",               default=data_para["V"], type=int)
parser.add_argument("--K",               default=data_para["K"], type=int)
parser.add_argument("--N",               default=data_para["N"], type=int)
parser.add_argument("--view_dims",       default=data_para["n_input"], type=int, nargs="+")

args = parser.parse_args()


def _normalize_para_loss(values):
    if len(values) == 6:
        return list(values)
    raise ValueError(
        "--para_loss expects exactly 6 values: "
        "[alpha, beta, eta, lambda, gamma, w_neigh]"
    )


args.para_loss = _normalize_para_loss(args.para_loss)


def main():
    seed_everything(args.seed)
    if not getattr(args, "quiet", False):
        print("--------missrate: {}--------".format(args.missrate))
        print(f"Dataset [{dataset_name}]: feature_dim={args.feature_dim}, dc={args.d_c}")

    X, Y, missindex, X_com, Y_com, _, _ = loader.load_data(args.dataset, args.missrate)

    loaded_views = len(X)
    if args.V > loaded_views:
        raise ValueError(f"--V={args.V} exceeds loaded views={loaded_views}.")
    if len(args.view_dims) < args.V:
        raise ValueError(f"--view_dims length={len(args.view_dims)} is smaller than --V={args.V}.")
    if len(args.view_dims) != args.V:
        args.view_dims = list(args.view_dims[: args.V])

    # Support sub-view experiments (e.g., Reuters 5->4/3/2 views) safely.
    if args.V < loaded_views:
        X = X[: args.V]
        Y = Y[: args.V]
        X_com = X_com[: args.V]
        Y_com = Y_com[: args.V]
        missindex = missindex[:, : args.V]

    miss_vecs = [missindex[:, v] for v in range(args.V)]

    model             = Network(args.V, args.view_dims, args.feature_dim).to(device)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.lr_pre)

    print_training_args(args, "Run")
    fea_pre = pretrain(model, optimizer_pretrain, args, device, X_com, Y_com, X, Y)

    proto_learner = ExactCayleyPrototypeLearner(
        views=args.V, feature_dim=args.feature_dim, num_clusters=args.K,
        dc_dim=args.d_c, pretrained_features=fea_pre
    ).to(device)

    optimizer_align = torch.optim.Adam(
        list(model.parameters()) + list(proto_learner.parameters()),
        lr=args.lr_finetuning
    )

    fea_end = train_align(
        model, proto_learner, optimizer_align,
        args, device, X, Y, miss_vecs, missindex
    )
    fea_end = [f.cpu().numpy() for f in fea_end]

    labels      = Y[0]
    fea_cluster = fea_end[0]
    for i in range(1, len(fea_end)):
        fea_cluster = np.concatenate((fea_cluster, fea_end[i]), axis=1)

  
    fea_cluster = np.nan_to_num(fea_cluster, nan=0.0, posinf=1e6, neginf=-1e6)

    estimator = KMeans(n_clusters=args.K)
    estimator.fit(fea_cluster)
    pred_final = estimator.labels_

    acc, nmi, purity, fscore, precision, recall, ari = evaluate(labels, pred_final)
    if getattr(args, "quiet", False):
        print(" ACC  |   NMI   |   F-Score  |    ARI   | Recall   |Precision")
        print(
            "{:.2f} |  {:.2f}  |    {:.2f}   |   {:.2f}  |   {:.2f}  |   {:.2f}  ".format(
                acc * 100,
                nmi * 100,
                fscore * 100,
                ari * 100,
                recall * 100,
                precision * 100,
            )
        )
    else:
        print("------------------------------------------RESULT END-------------------------------------------")
        print(" ACC  |   NMI   |   F-Score  |    ARI   | Recall   |Precision")
        print(
            "{:.2f} |  {:.2f}  |    {:.2f}   |   {:.2f}  |   {:.2f}  |   {:.2f}  ".format(
                acc * 100,
                nmi * 100,
                fscore * 100,
                ari * 100,
                recall * 100,
                precision * 100,
            )
        )
        print("-----------------------------------------------------------------------------------------------")



if __name__ == "__main__":
    main()
