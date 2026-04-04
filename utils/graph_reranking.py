import numpy as np
import torch
# import gc
import torch.nn.functional as F
import torch.nn as nn
import os

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
          
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def build_global_graph(probFea, galFea, k=20, gamma=0.5):
    """
    Xây dựng Global Graph (Equation 1 áp dụng cho toàn bộ)
    """
    features = torch.cat([probFea, galFea])
    features = F.normalize(features, p=2, dim=1)

    num_feature = features.size(0)
    dist = torch.cdist(features, features, p=2)
    topk_dist, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)

    weights = torch.exp(-(topk_dist ** 2)/ gamma)

    A = torch.zeros((num_feature, num_feature), device=probFea.device)

    for i in range(num_feature):
        weight_value = weights[i, :k]
        s = torch.sum(weight_value)
        if s > 0:
            A[i, topk_indices[i, :k]] = (weight_value / s).float()

    return A

def build_cross_camera_graph(probFea, galFea, q_camids, g_camids, k=20, gamma=0.5):
    """
    Xây dựng Cross-Camera Graph (Equation 1 có điều kiện camera khác nhau)
    """
    features = torch.cat([probFea, galFea])
    camids = torch.cat([q_camids, g_camids])

    num_feature = features.size(0)
    dist = torch.cdist(features, features, p=2)
    col = camids.unsqueeze(1)
    row = camids.unsqueeze(0)
    masked_id = (col != row)

    dist[~masked_id] = float('inf')
    topk_dist, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)

    weights = torch.exp(-(topk_dist ** 2)/ gamma)

    A = torch.zeros((num_feature, num_feature), device=probFea.device)


    for i in range(num_feature):
        weight_value = weights[i, :k]
        s = torch.sum(weight_value)
        if s > 0:
            A[i, topk_indices[i, :k]] = (weight_value / s).float()
        # else: all-inf distances (same-camera only dataset) → leave as zeros

    return A

def normalize_adj(adj_matrix):
    """
    Thực hiện chuẩn hóa: D^(-1/2) * A * D^(-1/2)
    """
    identity_matrix = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
    adj_matrix_hat = adj_matrix + identity_matrix

    D = adj_matrix_hat.sum(dim=1)
    # Add epsilon to avoid division by zero
    D = D + 1e-12
    d_inv_sqrt = torch.pow(D, -0.5)
    # Handle any inf or nan values
    d_inv_sqrt = torch.where(torch.isfinite(d_inv_sqrt), d_inv_sqrt, torch.zeros_like(d_inv_sqrt))
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    return d_mat_inv_sqrt @ adj_matrix_hat @ d_mat_inv_sqrt

def safe_to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return torch.stack(data).squeeze().to(device)
        else:
            return torch.tensor(data, device=device)
            
    if isinstance(data, np.ndarray):
        return torch.tensor(data, device=device)
        
    return torch.tensor(data, device=device)

def graph_reranking_func(probFea, galFea, q_camids, g_camids, k=20, gamma=0.5, alpha=0.8, learn_based=False, gcn_model=None):
    """
    Hàm chính gọi từ bên ngoài (Main entry point)
    
    Args:
        probFea: Tensor (N, D) - đặc trưng của query
        galFea: Tensor (M, D) - đặc trưng của gallery
        q_camids: Tensor (N,) - ID camera tương ứng của query
        g_camids: Tensor (N,) - ID camera tương ứng của gallery
    
    Returns:
        refined_distmat: numpy array (N, M) - ma trận khoảng cách đã được làm mịn
    """
    if k is None: k = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    q_camids = safe_to_tensor(q_camids, device=device)
    g_camids = safe_to_tensor(g_camids, device=device)
    
    # Normalize features first
    probFea = F.normalize(probFea, p=2, dim=1)
    galFea = F.normalize(galFea, p=2, dim=1)

    A_global = build_global_graph(probFea, galFea, k, gamma).to(device)
    A_cross = build_cross_camera_graph(probFea, galFea, q_camids, g_camids, k, gamma).to(device)
    
    A_global_norm = normalize_adj(A_global)
    # If A_cross is all-zeros or NaN (e.g. single-camera dataset like VehicleID),
    # keep it zero to match training behaviour instead of letting normalize_adj
    # turn it into identity via self-loop addition.
    if A_cross.sum() == 0 or torch.isnan(A_cross).any():
        A_cross_norm = torch.zeros_like(A_cross)
    else:
        A_cross_norm = normalize_adj(A_cross)
    
    # Compute initial distance matrix
    num_query = probFea.size(0)
    num_gallery = galFea.size(0)
    
    probFea = probFea.to(device).float()
    galFea = galFea.to(device).float()
    
    # Initial distance matrix (query x gallery)
    original_distmat = torch.cdist(probFea, galFea, p=2)
    
    # Create full distance matrix for graph propagation
    features = torch.cat([probFea, galFea], dim=0)
    full_distmat = torch.cdist(features, features, p=2)
    
    if gcn_model is not None:
        print("Using GCN model for graph re-ranking")
        gcn_model.eval()
        gcn_model = gcn_model.to(device)
        global_dim = gcn_model.W.shape[0]

        if features.shape[1] > global_dim:
            feat_global = features[:, :global_dim] 
            feat_local  = features[:, global_dim:]

            feat_global_refined = gcn_model(feat_global, A_global_norm, A_cross_norm)  

            feat_global_refined = F.normalize(feat_global_refined, p=2, dim=1)        
            feat_local = F.normalize(feat_local, p=2, dim=1) 

            refined_features = torch.cat((feat_global_refined, feat_local), dim=1)
        else:
            refined_features = gcn_model(features, A_global_norm, A_cross_norm)
            refined_features = F.normalize(refined_features, p=2, dim=1)
        
        # Compute distance from refined features
        refined_prob = refined_features[:num_query]
        refined_gal = refined_features[num_query:]
        distmat = torch.cdist(refined_prob, refined_gal, p=2)
    else:
        # Traditional graph propagation - refine distance matrix directly
        # Apply graph smoothing to distance matrix
        refined_distmat = alpha * torch.mm(A_global_norm, full_distmat) + \
                          (1 - alpha) * torch.mm(A_cross_norm, full_distmat)
        
        # Extract query-gallery distances
        distmat = refined_distmat[:num_query, num_query:]
           
    return distmat.detach().cpu().numpy() 

def build_graphs_for_batch(feat, camids, k=20, gamma=2.0):
    """
    Xây dựng đồ thị động cho một Batch huấn luyện.
    feat: (N, D) - Đặc trưng trích xuất từ Backbone
    camids: (N,) - ID camera của từng ảnh trong batch
    k: Số lượng lân cận gần nhất
    """
    # feat = torch.Tensor(feat)
    N = feat.size(0)
    device = feat.device

    dist_mat = torch.cdist(feat.float(), feat.float(), p=2)

    curr_k = min(k, N)
    _, nn_idx = torch.topk(dist_mat, k=curr_k, largest=False)
    
    A_g = torch.zeros(N, N, device=device)
    weights = torch.exp(-dist_mat / gamma)

    src_values = weights.gather(1, nn_idx)
    A_g.scatter_(1, nn_idx, src_values.to(A_g.dtype))

    camids = camids.to(device)
    camids = camids.view(-1, 1)
    mask_cross = (camids != camids.T).float()

    A_c = A_g * mask_cross

    return A_g, A_c

class GCNRefiner(nn.Module):
    def __init__(self, feature_dim):
        super(GCNRefiner, self).__init__()

        self.W = nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
        nn.init.kaiming_uniform_(self.W, a=0.2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.bn = nn.LayerNorm(feature_dim)
        self.relu = nn.ReLU()

    def forward(self, features, A_global_norm, A_cross_norm):
        # Ensure all inputs are float32 for stability
        features = features.float()
        A_global_norm = A_global_norm.float()
        A_cross_norm = A_cross_norm.float()      
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        support = alpha * torch.mm(A_global_norm, features) + \
                  (1 - alpha) * torch.mm(A_cross_norm, features)

        output_gcn = torch.mm(support, self.W)

        output_gcn = self.bn(output_gcn)
        output_gcn = self.relu(output_gcn)

        final_output = features + output_gcn
        
        return final_output
    
    def load_param(self, trained_path):
        if os.path.exists(trained_path):
            state_dict = torch.load(trained_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.load_state_dict(state_dict)
            print(f"==> GCN model loaded from {trained_path}")
        else:
            print(f"==> No GCN checkpoint found at {trained_path}, training from scratch.")