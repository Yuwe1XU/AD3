import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from utils.cython_merge.cython_merge import merge_cython
from utils.tsp_merge import merge as merge_cython_v2
from utils.tsp_merge import merge_wedges
from utils.tsp_merge_full import compute_distance_matrix as compute_distance_matrix_C
from utils.tsp_merge_full import merge_tsp_single
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -30:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  return tour, iterator


def numpy_merge(points, adj_mat):
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)
  merge_iterations = 0
  for edge in (-adj_mat / dists).flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
    real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
        ).toarray() for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          cython_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations
  

def merge_tours_v2(adj_mat, np_points, merge_thr = 1000, parallel_sampling=1):
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
  splitted_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat]
  splitted_points = [ np_points for _ in range(parallel_sampling) ]

  dist = compute_distance_matrix(splitted_points[0]) #np.linalg.norm(splitted_points[:,None,:] - splitted_points[None,:,:], axis=-1)
  start_time = time.time() 
  results = [ merge_cython_v2(_adj_mat, dist, merge_thr) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)]
  # print("Core Merge_pybind Time", time.time() - start_time)

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tour = [0]
  while len(tour) < splitted_adj_mat[0].shape[0] + 1:
    n = np.nonzero(splitted_real_adj_mat[0][tour[-1]])[0]
    if len(tour) > 1:  n = n[n != tour[-2]]
    tour.append(n.max())

  merge_iterations = np.mean(splitted_merge_iterations)
  return [tour], merge_iterations


def merge_tours_v3(adj_mat, np_points, merge_thr = 1000, parallel_sampling=1):
  adj_mat = adj_mat[0]
  splitted_adj_mat = adj_mat + adj_mat.T
  dist_mat = compute_distance_matrix(np_points) 

  start_time = time.time() 
  
  N = adj_mat.shape[0]
  # 上三角索引（不含对角线）
  i_idx, j_idx = np.triu_indices(N, k=1)
  # 对应距离
  dist_vals = dist_mat[i_idx, j_idx]
  # 过滤距离大于 threshold 的边
  # mask = dist_vals < merge_thr
  mask_threshold = dist_vals < merge_thr
  mask_random = np.random.rand(len(dist_vals)) < 0.1
  mask = mask_threshold | mask_random
  
  i_idx, j_idx = i_idx[mask], j_idx[mask]
  dist_vals = dist_vals[mask]
  # 对称矩阵 ⇒ (i,j) 与 (j,i) 都应保留（构建TSP边）
  i_all = np.concatenate([i_idx, j_idx])
  j_all = np.concatenate([j_idx, i_idx])
  dist_all = np.concatenate([dist_vals, dist_vals])
  # 计算对应的 adj 和权重
  adj_vals = adj_mat[i_all, j_all]
  weights = adj_vals / dist_all  # w = adj / d
  # 计算 flat indices (i*N + j)
  flat_indices = i_all * N + j_all
  # 排序（按 -w 作为优先级）
  order = np.argsort(-weights)  # 直接按降序排，不用 -w 存储再升序排

  flat_indices = flat_indices[order].astype(np.int32)

  print("Insert&Sort Cost Time", time.time() - start_time)

  results = [merge_with_flat_indices_py(N, flat_indices)]
  print("Core Merge_pybind Time", time.time() - start_time)

  tour, merge_iterations = zip(*results)

  return tour, merge_iterations



def merge_tours_v4(adj_mat, np_points, merge_thr = 1000, parallel_sampling=1):
  dist_mat = compute_distance_matrix_C(np_points) 
  tour, iterations = merge_tsp_single(adj_mat[0], dist_mat, merge_thr)

  return [tour], iterations



def prepare_and_merge(adj_mat: np.ndarray, dist_mat: np.ndarray, thr: float = 1000.0):
    N = adj_mat.shape[0]
    assert adj_mat.shape == (N, N) and dist_mat.shape == (N, N)

    # 避免除以0或非常小的数（对角线）
    dist_safe = np.where(dist_mat < 1, 1, dist_mat)

    # 计算边的得分权重和采样概率
    weights = adj_mat / dist_safe
    probs = np.minimum(1.0, thr / dist_safe)

    # 生成所有候选边的 index 对 (i, j)
    i_idx, j_idx = np.triu_indices(N, k=1)  # 只取上三角，避免重复和自环

    # 筛选有效边
    sampled = np.random.rand(len(i_idx)) < probs[i_idx, j_idx]
    i_idx = i_idx[sampled]
    j_idx = j_idx[sampled]
    flat_idx = i_idx * N + j_idx
    edge_weights = weights[i_idx, j_idx]

    # 粗略排序（近似排序）
    sort_order = np.argsort(-edge_weights)  # 由大到小
    sorted_flat_idx = flat_idx[sort_order]

    # 调用 C++ merge 函数
    return merge_wedges(sorted_flat_idx, int(N))


def find_root(parents, x):
    while parents[x] != x:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x

def merge_with_flat_indices_py(N, flat_indices):
    route_begin = list(range(N))
    route_end = list(range(N))
    uf = list(range(N))
    print("  0")

    added_edges = []
    print("  01")

    merge_count = 0
    merge_iterations = 0
    M = flat_indices.shape[0]

    for ei in range(M):
        merge_iterations += 1
        flat = int(flat_indices[ei])
        i = flat // N
        j = flat % N

        bi = find_root(route_begin, i)
        ei_ = find_root(route_end, i)
        bj = find_root(route_begin, j)
        ej = find_root(route_end, j)

        if bi == bj:
            continue
        if i != bi and i != ei_:
            continue
        if j != bj and j != ej:
            continue

        added_edges.append((i, j))
        merge_count += 1

        if i == bi and j == ej:
            route_begin[bi] = bj
            route_end[ej] = ei_
        elif i == ei_ and j == bj:
            route_begin[bj] = bi
            route_end[ei_] = ej
        elif i == bi and j == bj:
            route_begin[bi] = ej
            route_begin[bj] = ej
            route_begin[ej] = ej
            route_end[ej] = ei_
            route_end[bj] = ei_
        elif i == ei_ and j == ej:
            route_end[ei_] = bj
            route_begin[bj] = bi
            route_begin[ej] = bi
            route_end[ej] = bj
            route_end[bj] = bj

        if merge_count == N - 1:
            break
    print("  1")

    fb = find_root(route_begin, 0)
    fe = find_root(route_end, 0)
    added_edges.append((fb, fe))
    merge_iterations += 1
    print("  2")

    adj = [[] for _ in range(N)]
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    print("  3")

    tour = []
    curr, prev = 0, -1
    for _ in range(N):
        tour.append(curr)
        next_ = adj[curr][1] if adj[curr][0] == prev else adj[curr][0]
        prev, curr = curr, next_
    tour.append(0)
    print("  4")

    print("  5")
    return np.array(tour, dtype=np.int32), merge_iterations



def compute_distance_matrix(points):
    """
    计算所有点对之间的欧氏距离矩阵。
    points: (n,2) 数组
    返回: (n,n) 矩阵
    """
    diff = points[:, None, :] - points[None, :, :]
    return np.hypot(diff[..., 0], diff[..., 1])


def two_opt_numpy(points, tour, max_iterations=1000, sample_size=200):
    """
    使用 numpy 加速的 2-opt 优化，采用随机采样 + 向量化。

    参数：
    - points: ndarray, shape (n,2)
    - tour: ndarray of ints, shape (n,), 表示访问顺序，首尾不重复
    - max_iterations: 最大迭代次数
    - sample_size: 每轮从所有可能的 i-j 对中采样的数量

    返回：
    - tour_opt: ndarray, shape (n,), 优化后路径
    - iterations: 实际迭代次数
    """
    n = len(tour)
    D = compute_distance_matrix(points)
    tour = tour.copy()
    iteration = 0

    while iteration < max_iterations:
        # 从所有可能的 i-j 对中采样
        i_vals = np.random.randint(1, n - 3, size=sample_size)
        j_vals = np.random.randint(i_vals + 2, n - 1)

        a = tour[i_vals - 1]
        b = tour[i_vals]
        c = tour[j_vals]
        d = tour[j_vals + 1]

        # 向量化计算 delta
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        best_idx = np.argmin(delta)

        if delta[best_idx] < 0:
            i = i_vals[best_idx]
            j = j_vals[best_idx]
            tour[i:j+1] = tour[i:j+1][::-1]
        else:
            break

        iteration += 1

    return tour, iteration

class TSPEvaluator(object):
  def __init__(self, points):
    self.dist_mat = scipy.spatial.distance_matrix(points, points)

  def evaluate(self, route):
    total_cost = 0
    for i in range(len(route) - 1):
      total_cost += self.dist_mat[route[i], route[i + 1]]
    return total_cost
