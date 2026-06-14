import numpy as np
import torch
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


def build_global_graph(probFea, galFea, lambda1=4.2, gamma=0.2):
    """
    Xây dựng Global Graph (Equation 1 áp dụng cho toàn bộ)
    """
    features = torch.cat([probFea, galFea])
    features = F.normalize(features, p=2, dim=1)

    num_feature = features.size(0)
    device = features.device
    dist = torch.cdist(features, features, p=2)
    # topk_dist, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)

    # Apply Cheb-GR in order to eliminate unnecessary edge in graph
    mean = torch.mean(dist, dim=1, keepdim=True)
    std = torch.std(dist, dim=1, keepdim=True)

    # Ti = µi − λσ
    threshold = mean - lambda1 * std
    dist_adjust = dist.clone()
    dist_adjust[dist_adjust > threshold] = float("inf")

    # Reliability-Aware Edge Weighting
    std_i, std_j = std, std.t()
    reliability_scale = std_i * std_j + 1e-8
    g_ij = torch.exp(-(dist ** 2) / reliability_scale)
    base_weights = torch.exp(-(dist_adjust ** 2)/ gamma)

    weights = base_weights * g_ij
    row_sum = weights.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1e-12
    norm_weights = weights / row_sum

    indices = torch.nonzero(weights > 0).t()
    values = norm_weights[indices[0], indices[1]]

    A = torch.sparse_coo_tensor(
        indices,
        values,
        (num_feature, num_feature),
        device=device
    ).coalesce()

    return A

def build_cross_camera_graph(probFea, galFea, q_camids, g_camids, lambda1=4.2, gamma=0.2):
    """
    Xây dựng Cross-Camera Graph (Equation 1 có điều kiện camera khác nhau)
    """
    features = torch.cat([probFea, galFea])
    camids = torch.cat([q_camids, g_camids])

    num_feature = features.size(0)
    device = features.device

    dist = torch.cdist(features, features, p=2)
    col = camids.unsqueeze(1)
    row = camids.unsqueeze(0)
    masked_id = (col != row)

    dist[~masked_id] = float('inf')

    # Apply Cheb-GR in order to eliminate unnecessary edge in graph
    mean = torch.mean(dist, dim=1, keepdim=True)
    std = torch.std(dist, dim=1, keepdim=True)

    # Ti = µi − λσ
    threshold = mean - lambda1 * std
    dist_adjust = dist.clone()
    dist_adjust[dist_adjust > threshold] = float("inf")

    # Reliability-Aware Edge Weighting
    std_i, std_j = std, std.t()
    reliability_scale = std_i * std_j + 1e-8
    g_ij = torch.exp(-(dist ** 2) / reliability_scale)
    base_weights = torch.exp(-(dist_adjust ** 2)/ gamma)

    weights = base_weights * g_ij
    row_sum = weights.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1e-12
    norm_weights = weights / row_sum

    indices = torch.nonzero(weights > 0).t()
    values = norm_weights[indices[0], indices[1]]

    A = torch.sparse_coo_tensor(
        indices,
        values,
        (num_feature, num_feature),
        device=device
    ).coalesce()

    return A

def normalize_adj(adj_matrix):
    """
    Thực hiện chuẩn hóa đối xứng ma trận kề: D^(-1/2) * A_hat * D^(-1/2)
    Hỗ trợ tự động cả ma trận dạng Đặc (Dense) khi Train và dạng Thưa (Sparse) khi Test.
    """
    device = adj_matrix.device
    n = adj_matrix.shape[0]

    # Kiểm tra xem ma trận kề là dạng Thưa (Sparse) hay dạng Đặc (Dense)
    if adj_matrix.is_sparse:
        # ----------------------------------------------------
        # PHÂN NHÁNH 1: CHUẨN HÓA CHO MA TRẬN THƯA (Pha Test)
        # ----------------------------------------------------
        identity = torch.eye(n, device=device).to_sparse()
        adj_hat = (adj_matrix + identity).coalesce()

        D = torch.sparse.sum(adj_hat, dim=1).to_dense()
        D = D + 1e-12
        d_inv_sqrt = torch.pow(D, -0.5)
        d_inv_sqrt = torch.where(torch.isfinite(d_inv_sqrt), d_inv_sqrt, torch.zeros_like(d_inv_sqrt))

        indices = adj_hat.indices()
        values = adj_hat.values()

        row_scale = d_inv_sqrt[indices[0]]
        col_scale = d_inv_sqrt[indices[1]]
        scaled_values = values * row_scale * col_scale

        return torch.sparse_coo_tensor(indices, scaled_values, (n, n), device=device).coalesce()
    else:
        # ----------------------------------------------------
        # PHÂN NHÁNH 2: CHUẨN HÓA CHO MA TRẬN ĐẶC (Pha Train)
        # ----------------------------------------------------
        # Thêm self-loop bằng ma trận đơn vị đặc
        identity = torch.eye(n, device=device)
        adj_hat = adj_matrix + identity

        # Tính toán ma trận bậc D (tổng hàng)
        D = adj_hat.sum(dim=1)
        D = D + 1e-12
        d_inv_sqrt = torch.pow(D, -0.5)
        d_inv_sqrt = torch.where(torch.isfinite(d_inv_sqrt), d_inv_sqrt, torch.zeros_like(d_inv_sqrt))

        # Sử dụng cơ chế broadcasting để thực hiện phép nhân đối xứng: d_i * A_ij * d_j
        # Phép toán này có độ phức tạp chỉ O(N^2), nhanh hơn nhiều so với nhân ma trận thông thường
        norm_adj = d_inv_sqrt.view(-1, 1) * adj_hat * d_inv_sqrt.view(1, -1)
        return norm_adj

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

def build_graphs_for_batch(feat, camids, lambda1=4.2, gamma=0.2):
    """
    Xây dựng đồ thị động cho một Batch huấn luyện.
    feat: (N, D) - Đặc trưng trích xuất từ Backbone
    camids: (N,) - ID camera của từng ảnh trong batch
    k: Số lượng lân cận gần nhất
    """
    num_feature = feat.size(0)
    device = feat.device

    dist = torch.cdist(feat.float(), feat.float(), p=2)

    ### BUILD GLOBAL GRAPH
    # Apply Cheb-GR in order to eliminate unnecessary edge in graph
    mean_g = torch.mean(dist, dim=1, keepdim=True)
    variance_g = torch.mean((dist - mean_g) ** 2, dim=1, keepdim=True)
    std_g = torch.sqrt(variance_g + 1e-8)

    # Ti = µi − λσ
    threshold_g = mean_g - lambda1 * std_g
    valid_mask_g = (dist <= threshold_g)

    # Reliability-Aware Edge Weighting
    std_i_g, std_j_g = std_g, std_g.t()
    reliability_scale_g = std_i_g * std_j_g + 1e-8
    g_ij_g = torch.exp(-(dist ** 2) / reliability_scale_g)

    base_weights_g = torch.exp(-(dist ** 2)/ gamma)
    weights_g = base_weights_g * g_ij_g
    weights_g = torch.where(valid_mask_g, weights_g, torch.zeros_like(weights_g))

    row_sum_g = weights_g.sum(dim=1, keepdim=True)
    row_sum_g[row_sum_g == 0] = 1e-12
    
    row_sum_g = weights_g.sum(dim=1, keepdim=True)
    A_g = weights_g / (row_sum_g + 1e-12)

    ### BUILD CROSS GRAPH
    camids = camids.to(device).view(-1, 1)
    mask_cross = (camids != camids.T).float()

    num_cross = mask_cross.sum(dim=1, keepdim=True)

    # Apply Cheb-GR in order to eliminate unnecessary edge in graph
    mean_c = (dist * mask_cross).sum(dim=1, keepdim=True) / (num_cross + 1e-8)
    variance_c = (((dist - mean_c) ** 2) * mask_cross).sum(dim=1, keepdim=True) / (num_cross + 1e-8)
    std_c = torch.sqrt(variance_c + 1e-8)

    threshold_c = mean_c - lambda1 * std_c
    valid_mask_c = (mask_cross == 1) & (dist <= threshold_c)

    # Reliability-Aware Edge Weighting
    std_i_c, std_j_c = std_c, std_c.t()
    reliability_scale_c = std_i_c * std_j_c + 1e-8
    g_ij_c = torch.exp(-(dist ** 2) / reliability_scale_c)

    base_weights_c = torch.exp(-(dist ** 2) / gamma)
    weights_c = base_weights_c * g_ij_c
    weights_c = torch.where(valid_mask_c, weights_c, torch.zeros_like(weights_c))

    row_sum_c = weights_c.sum(dim=1, keepdim=True)
    A_c = weights_c / (row_sum_c + 1e-12)

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