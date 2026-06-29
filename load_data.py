import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import scipy.io
import scipy.sparse as sp
import random
import warnings

warnings.filterwarnings("ignore")


ALL_data = dict(
    Caltech101_7={1: "Caltech101_7", "N": 1400, "K": 7, "V": 5, "n_input": [1984, 512, 928, 254, 40]},
    HandWritten={1: "handwritten1031_v73", "N": 2000, "K": 10, "V": 6, "n_input": [240, 76, 216, 47, 64, 6]},
    Caltech101_20={1: "Caltech101-20", "N": 2386, "K": 20, "V": 6, "n_input": [48, 40, 254, 1984, 512, 928]},
    LandUse_21={1: "LandUse_21_v73", "N": 2100, "K": 21, "V": 3, "n_input": [20, 59, 40]},
    Scene_15={1: "Scene_15_v73", "N": 4485, "K": 15, "V": 3, "n_input": [20, 59, 40]},
    ALOI_100={1: "ALOI_100_7", "N": 10800, "K": 100, "V": 4, "n_input": [77, 13, 64, 125]},
    YouTubeFace10_4Views={1: "YTF10_4", "N": 38654, "K": 10, "V": 4, "n_input": [944, 576, 512, 640]},
    AWA={1: "AWA_73", "N": 10158, "K": 50, "V": 7, "n_input": [2688, 2000, 2000, 2000, 2000, 4096, 4096]},
    EMNIST_digits_4Views={1: "EMNIST_digits_4Views_v73", "N": 280000, "K": 10, "V": 4, "n_input": [944, 576, 512, 640]},
    Leaves100={1: "100leaves", "N": 1600, "K": 100, "V": 3, "n_input": [64, 64, 64]},
    News20={1: "20newsgroups", "N": 500, "K": 5, "V": 3, "n_input": [2000, 2000, 2000]},
    Sources3={1: "3sources", "N": 169, "K": 6, "V": 3, "n_input": [3068, 3631, 3560]},
    ACM3025={1: "ACM3025", "N": 3025, "K": 3, "V": 1, "n_input": [1870]},
    Animal={1: "animal", "N": 10158, "K": 50, "V": 2, "n_input": [4096, 4096]},
    BBC={1: "BBC", "N": 685, "K": 5, "V": 4, "n_input": [4659, 4633, 4665, 4684]},
    BBCSport={1: "bbcsport", "N": 544, "K": 5, "V": 2, "n_input": [3183, 3203]},
    BBCSport_2View1={1: "bbcsport_2view1", "N": 544, "K": 5, "V": 2, "n_input": [3183, 3203]},
    Caltech101_7_AllViews={1: "Caltech101-7", "N": 1474, "K": 7, "V": 6, "n_input": [48, 40, 254, 1984, 512, 928]},
    COIL100={1: "COIL100", "N": 7200, "K": 100, "V": 1, "n_input": [1024]},
    COIL20={1: "COIL20", "N": 1440, "K": 20, "V": 3, "n_input": [1024, 3304, 6750]},
    CiteSeer={1: "CiteSeer", "N": 3312, "K": 6, "V": 2, "n_input": [3312, 3703]},
    CUB_GoogLeNet_Doc2Vec_C10={1: "cub_googlenet_doc2vec_c10", "N": 600, "K": 10, "V": 2, "n_input": [1024, 300]},
    Digit4k={1: "Digit4k", "N": 4000, "K": 4, "V": 2, "n_input": [30, 30]},
    ESP_Game={1: "esp_game", "N": 11032, "K": 7, "V": 2, "n_input": [100, 100]},
    Flower17={1: "flower17", "N": 1360, "K": 17, "V": 7, "n_input": [1360, 1360, 1360, 1360, 1360, 1360, 1360]},
    Food_101={1: "Food-101", "N": 86796, "K": 101, "V": 3, "n_input": [768, 512, 512]},
    HDigit={1: "Hdigit", "N": 10000, "K": 10, "V": 2, "n_input": [784, 256]},
    Mfeat={1: "Mfeat", "N": 2000, "K": 10, "V": 6, "n_input": [216, 76, 64, 6, 240, 47]},
    MNIST_3Views={1: "MNIST", "N": 10000, "K": 10, "V": 3, "n_input": [30, 9, 30]},
    MSRC_v6={1: "MSRC-v6", "N": 210, "K": 7, "V": 6, "n_input": [1302, 48, 512, 100, 256, 210]},
    MSRC={1: "MSRC", "N": 210, "K": 7, "V": 5, "n_input": [24, 576, 512, 256, 254]},
    MSRC_v1={1: "MSRC_v1", "N": 210, "K": 7, "V": 5, "n_input": [24, 576, 512, 256, 254]},
    NUS_WIDE={1: "NUS_WIDE", "N": 2000, "K": 31, "V": 5, "n_input": [65, 226, 145, 74, 129]},
    ORL={1: "ORL", "N": 400, "K": 40, "V": 3, "n_input": [4096, 3304, 6750]},
    PIE_face_10={1: "PIE_face_10", "N": 680, "K": 68, "V": 3, "n_input": [484, 256, 279]},
    Prokaryotic={1: "prokaryotic", "N": 551, "K": 4, "V": 1, "n_input": [438]},
    Reuters={1: "Reuters", "N": 1200, "K": 6, "V": 5, "n_input": [2000, 2000, 2000, 2000, 2000]},
    Reuters_21578={1: "Reuters_21578", "N": 1500, "K": 6, "V": 5, "n_input": [21531, 24892, 34251, 15506, 11547]},
    ThreeRing={1: "ThreeRing", "N": 300, "K": 3, "V": 2, "n_input": [2, 2]},
    TwoMoon={1: "TwoMoon", "N": 200, "K": 2, "V": 2, "n_input": [2, 2]},
    UCI_Digit6={1: "UCI_Digit6", "N": 2000, "K": 10, "V": 6, "n_input": [216, 76, 64, 240, 47, 6]},
    USPS={1: "USPS", "N": 9298, "K": 10, "V": 1, "n_input": [256]},
    VGGFace2_50={1: "VGGFace2-50", "N": 34027, "K": 50, "V": 4, "n_input": [944, 576, 512, 640]},
    WebKB={1: "WebKB", "N": 203, "K": 4, "V": 3, "n_input": [1703, 230, 230]},
    WikipediaArticles={1: "WikipediaArticles", "N": 693, "K": 10, "V": 2, "n_input": [128, 10]},
    Yale={1: "yale", "N": 165, "K": 15, "V": 3, "n_input": [4096, 3304, 6750]},
    YaleB={1: "YaleB", "N": 650, "K": 10, "V": 3, "n_input": [2500, 3304, 6750]},
    Youtube={1: "Youtube", "N": 2000, "K": 10, "V": 6, "n_input": [2000, 1024, 64, 512, 64, 647]},
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(_THIS_DIR, "Dataset") + os.sep

# If the default stem is missing, try these basenames (no ".mat") in order.
_MAT_STEM_ALIASES = {
    "Caltech101-20": [
        "Caltech101-20",
        "Caltech101-20_v73",
        "Caltech101_20",
    ],
    "CiteSeer": [
        "CiteSeer",
        "Citeseer",
        "citeseer",
    ],
    "Reuters": [
        "Reuters",
        "reuters",
    ],
}


def _resolve_mat_filepath(stem: str) -> str:
    candidates = _MAT_STEM_ALIASES.get(stem, [stem])
    tried = []
    for name in candidates:
        fp = path + name + ".mat"
        tried.append(os.path.abspath(fp))
        if os.path.isfile(fp):
            return fp
    raise FileNotFoundError(
        "Dataset .mat not found. Tried:\n  "
        + "\n  ".join(tried)
        + f"\nPut the file in: {os.path.abspath(path)}"
    )


def get_mask(view_num, alldata_len, missing_rate):
    missindex = np.ones((alldata_len, view_num))
    miss_begin = int((1 - missing_rate) * alldata_len)
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num, size=view_num - 1)
        missindex[i, missdata] = 0
    return missindex


def Form_Incomplete_Data(missrate=0.5, X=None, Y=None):
    if X is None:
        X = []
    if Y is None:
        Y = []

    np.random.seed(1)
    size = len(Y[0])
    view_num = len(X)
    index = list(range(size))
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    missindex = get_mask(view_num, size, missrate)
    index_complete, index_partial = [[] for _ in range(view_num)], [[] for _ in range(view_num)]
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index_complete[j].append(i)
            else:
                index_partial[j].append(i)

    max_len = max([len(idx) for idx in index_complete])
    filled_index_com = [[] for _ in range(view_num)]
    for v in range(view_num):
        if len(index_complete[v]) < max_len:
            diff_len = max_len - len(index_complete[v])
            if len(index_complete[v]) == 0:
                diff_value = random.choices(list(range(size)), k=diff_len)
                filled_index_com[v] = diff_value
            else:
                diff_value = random.choices(index_complete[v], k=diff_len)
                filled_index_com[v] = index_complete[v] + diff_value
        else:
            filled_index_com[v] = index_complete[v]

    filled_X_complete, filled_Y_complete = [[] for _ in range(view_num)], [[] for _ in range(view_num)]
    for i in range(view_num):
        filled_X_complete[i] = X[i][filled_index_com[i]]
        filled_Y_complete[i] = Y[i][filled_index_com[i]]
    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])
        filled_X_complete[v] = torch.from_numpy(filled_X_complete[v])

    return X, Y, missindex, filled_X_complete, filled_Y_complete, index_complete, index_partial


def load_data(dataset, missrate):
    filepath = _resolve_mat_filepath(dataset[1])
    X, Y = [], []
    mm = MinMaxScaler()

    def _to_dense_view(arr_like):
        arr = arr_like
        while isinstance(arr, np.ndarray) and arr.dtype == object:
            if arr.size == 0:
                return None
            arr = arr.reshape(-1)[0]
        if sp.issparse(arr):
            arr = arr.toarray()
        arr = np.array(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _extract_label_np(data_dict):
        for key in ("Y", "y", "gt", "gnd", "label", "labels", "truelabel", "y0"):
            if key in data_dict:
                lb = np.array(data_dict[key])
                if lb.dtype == object and lb.size > 0:
                    lb = np.array(lb.reshape(-1)[0])
                if lb.ndim > 1 and lb.shape[1] > 1:
                    if key == "y0" and np.all(lb == lb[:, [0]]):
                        lb = lb[:, 0]
                    elif key in ("label", "labels"):
                        lb = np.argmax(lb, axis=1) + 1
                return lb.reshape(-1)
        raise KeyError("No label key found in mat file.")

    def _append_view(view_np, label_np):
        if view_np is None:
            return
        if view_np.shape[0] != label_np.shape[0] and view_np.shape[1] == label_np.shape[0]:
            view_np = view_np.T
        std_view = mm.fit_transform(view_np)
        X.append(std_view)
        Y.append(label_np)

    try:
        data = h5py.File(filepath, 'r')
        try:
            label = _extract_label_np(data)
            if "X" in data:
                refs = np.array(data["X"]).reshape(-1)
                for ref in refs:
                    diff_view = _to_dense_view(np.array(data[ref]))
                    _append_view(diff_view.T, label)
            else:
                for key in ("image_features", "class_text_features", "desc_text_features", "feature"):
                    if key in data:
                        diff_view = _to_dense_view(np.array(data[key]))
                        _append_view(diff_view, label)
        finally:
            data.close()
    except OSError:
        data = scipy.io.loadmat(filepath)
        label = _extract_label_np(data)
        view_keys = ("X", "fea", "data")
        chosen_view_key = None
        for key in view_keys:
            if key in data:
                chosen_view_key = key
                break

        if chosen_view_key is not None:
            x_cell = data[chosen_view_key]
            if isinstance(x_cell, np.ndarray) and x_cell.dtype == object:
                for cell in x_cell.reshape(-1):
                    diff_view = _to_dense_view(cell)
                    _append_view(diff_view, label)
            else:
                diff_view = _to_dense_view(x_cell)
                _append_view(diff_view, label)
        else:
            for key in ("image_features", "class_text_features", "desc_text_features", "feature"):
                if key in data:
                    diff_view = _to_dense_view(data[key])
                    _append_view(diff_view, label)

    if len(X) == 0:
        raise ValueError(f"No valid views loaded from: {filepath}")

    return Form_Incomplete_Data(missrate=missrate, X=X, Y=Y)
