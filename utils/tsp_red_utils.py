import os
import elkai
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from co_datasets.tsp_graph_dataset_old import TSPGraphDataset
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
# from ortools.graph import pywrapgraph  # 需要安装ortools
from ortools.graph import pywrapgraph
import heapq
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours, merge_tours_v2, merge_tours_v4, two_opt_numpy

from tqdm import tqdm
import argparse
import logging, time
# import utils.tsp_optimizer as tsp_optimizer

RANDOM_SEED = 42

def setup_logging(args, log_file, log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # add on mode
            # logging.StreamHandler()  # output to terminal
        ]
    )
    logging.info("-" * 100)
    logging.info("Program started with the following parameters:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def build_coord_dict(points, prefix=""):
    return {f"{prefix}_{i}": (float(x), float(y)) for i, (x, y) in enumerate(points)}

def plot_global(points, labels, tour_indices, centers = None, super_tour = None, save_path = "./figures/global_tsp_result.png"):
    plt.figure(figsize=(25, 25))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:16]  #
    for i in range(len(centers)):
        mask = (labels == i)
        plt.scatter(points[mask, 0], points[mask, 1],  color=colors[i % len(colors)], s=10, alpha=0.3, zorder=0)  #

        plt.text(centers[i, 0], centers[i, 1], str(i), 
                fontsize=28, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=20)
     
   
    if super_tour is not None:
        super_points = np.array([centers[i] for i in super_tour])
        plt.plot(super_points[:,0], super_points[:,1], 'k--', alpha=0.5, zorder=5)
        plt.savefig("./figures/global_super_tsp_result.png", dpi=300)
    
    ordered_points = points[tour_indices]
    plt.plot(ordered_points[:,0], ordered_points[:,1], 'r-', linewidth=0.5, alpha=0.7, zorder=10)
    plt.title("Global TSP Path with Cluster Coloring")
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_clusters(points, labels, centers):
    k = len(np.unique(labels))
    plt.figure(figsize=(25, 25))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for i in range(k):
        cluster_points = points[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i%20], s=10, alpha=0.3, zorder=0)
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Centers')

    plt.title("Balanced Clusters via K-Means + MCMF")
    plt.savefig("./figures/clusters.png", dpi=300)
    plt.show()

def plot_cluster_tour(enhanced_points, main_points, tour, cluster_id, output_dir = './figures/tsp_90subgraph/'):
    if output_dir is not None: 
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.scatter(enhanced_points[:,0], enhanced_points[:,1], c='grey', s=20, alpha=0.5)
    plt.scatter(main_points[:,0], main_points[:,1], c='red', s=30, label='Main Cluster')
    plt.plot(enhanced_points[tour][:,0], enhanced_points[tour][:,1], 'b-', alpha=0.5)
    plt.title(f"Enhanced Subgraph #{cluster_id}\nTotal Nodes: {len(enhanced_points)}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"subgraph_{cluster_id}.png"), dpi=150)
    plt.close()


def plot_cluster_wotour(enhanced_points, main_points, cluster_id, output_dir = './figures/tsp_90subgraph/'):
    if output_dir is not None: 
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.scatter(enhanced_points[:,0], enhanced_points[:,1], c='grey', s=20, alpha=0.5)
    plt.scatter(main_points[:,0], main_points[:,1], c='red', s=30, label='Main Cluster')
    plt.title(f"Enhanced Subgraph #{cluster_id}\nTotal Nodes: {len(enhanced_points)}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"subgraph_{cluster_id}.png"), dpi=150)
    plt.close()


def balanced_kmeans_mcmf(points, k, m=3):
    n = len(points)
    if n % k != 0: raise ValueError("n must be divisible by k for Balanced Clustering.")
    cluster_size = n // k
    
    # Step 1: K-means initialization
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    initial_labels = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_
    
    # Step 2: Compute distances between points and cluster centers
    distances = cdist(points, centers)
    G = nx.DiGraph()

    source, sink = "source", "sink"
    G.add_node(source, demand=-n)
    G.add_node(sink, demand=n)
    
    # Add nodes for each point and each cluster
    for i in range(n): G.add_node(f"p{i}", demand=0)
    for j in range(k): G.add_node(f"c{j}", demand=0)
    # Edge from source to each point (capacity 1, no cost)
    for i in range(n): G.add_edge(source, f"p{i}", capacity=1, weight=0)
    
    # Edge from each point to candidate clusters
    for i in range(n):
        assigned_cluster = initial_labels[i]
        # Use m nearest clusters
        nearest_clusters = np.argsort(distances[i])[:m]
        candidate_clusters = set(nearest_clusters) | {assigned_cluster}
        for j in candidate_clusters:   
            cost = round(1000 * distances[i, j])  # Multiply by 1000 to convert to integer
            G.add_edge(f"p{i}", f"c{j}", capacity=1, weight=cost)
    
    # Edge from each cluster to sink, enforcing cluster_size
    for j in range(k):  G.add_edge(f"c{j}", sink, capacity=cluster_size, weight=0)
    

    # Step 3: Solve the min-cost flow problem using network simplex
    flowCost, flowDict = nx.network_simplex(G)
    
    # Step 4: Extract the assignment from the flow dictionary. the flow from point node p{i} , target is like "c{j}"
    final_labels = np.full(n, -1, dtype=int)
    for i in range(n):
        flows = flowDict[f"p{i}"]
        for target, flow in flows.items():
            if flow > 0 and target.startswith("c"):            
                final_labels[i] = int(target[1:])
                break
                
    return final_labels, centers




def balanced_kmeans_mcmf_fast(points, k, m=5):
    n = len(points)
    assert n % k == 0
    sz = n // k

    # 1) 初始化 KMeans
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(points)
    centers = km.cluster_centers_

    # 2) 计算距离 + 候选簇
    dist = pairwise_distances(points, centers)  # shape: (n, k)
    nearest_m = np.argpartition(dist, m, axis=1)[:, :m]
    assigned = km.labels_.reshape(-1, 1)
    candidates = np.concatenate([nearest_m, assigned], axis=1)

    # 3) 构建边：转成 Python int
    edges = []
    for i in range(n):  # source (0) -> points (2+i)
        edges.append((0, 2 + i, 1, 0))
    for i in range(n):  # point (2+i) -> cluster (2+n+j)
        for j in set(candidates[i]):
            cost = int(dist[i, j] * 1000)  # Python int
            edges.append((2 + i, 2 + n + j, 1, cost))
    for j in range(k):  # cluster -> sink (1)
        edges.append((2 + n + j, 1, sz, 0))

    # 4) 使用 OR-Tools 最小费用最大流
    smcf = pywrapgraph.SimpleMinCostFlow()
    for u, v, cap, cost in edges:
        smcf.AddArcWithCapacityAndUnitCost(int(u), int(v), int(cap), int(cost))
    smcf.SetNodeSupply(0, -int(n))
    smcf.SetNodeSupply(1, int(n))

    status = smcf.Solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError("Flow solver failed.")

    # 5) 提取分配结果
    labels = np.full(n, -1, dtype=int)
    for arc in range(smcf.NumArcs()):
        if smcf.Flow(arc) > 0:
            u = smcf.Tail(arc)
            v = smcf.Head(arc)
            if 2 <= u < 2 + n and v >= 2 + n:
                labels[u - 2] = v - (2 + n)

    return labels, centers

def balanced_kmeans_mcmf_fast_v3(points, k, m=5):
    n = len(points)
    if n % k != 0:   raise ValueError(f"数据点数量 {n} 必须能被聚类数量 {k} 整除")
    
    cluster_size = n // k
    
    # 1) K-means初始化
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    initial_labels = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_
    
    # 2) 计算距离矩阵
    distances = pairwise_distances(points, centers)
    
    # 3) 为每个点确定候选聚类
    candidates_per_point = []
    for i in range(n):
        nearest_clusters = np.argsort(distances[i])[:m]
        assigned_cluster = initial_labels[i]
        candidate_set = set(nearest_clusters) | {assigned_cluster}
        candidates_per_point.append(sorted(candidate_set))
    
    # 4) 构建最小费用最大流图
    smcf = pywrapgraph.SimpleMinCostFlow()
    
    # 添加边
    for i in range(n):
        point_node = int(2 + i)
        smcf.AddArcWithCapacityAndUnitCost(0, point_node, 1, 0)
    
    # 点到候选聚类的边
    for i in range(n):
        point_node = 2 + i
        for j in candidates_per_point[i]:
            cluster_node = int( 2 + n + j) 
            # 将距离转换为整数费用，乘以1000并四舍五入
            cost = int(round(distances[i, j]))
            smcf.AddArcWithCapacityAndUnitCost(point_node, cluster_node, 1, cost)
    
    # 聚类到sink的边 (容量=cluster_size, 费用=0)
    for j in range(k):
        cluster_node = int(2 + n + j)
        smcf.AddArcWithCapacityAndUnitCost(cluster_node, 1, cluster_size, 0)
    
    # 设置供需
    smcf.SetNodeSupply(0, n)      # source供应n个单位
    smcf.SetNodeSupply(1, -n)     # sink需求n个单位
    
    # 5) 求解
    status = smcf.Solve()
    
    # 检查求解状态
    status_names = {
        smcf.OPTIMAL: "OPTIMAL",
        smcf.NOT_SOLVED: "NOT_SOLVED", 
        smcf.FEASIBLE: "FEASIBLE",
        smcf.INFEASIBLE: "INFEASIBLE",
        smcf.UNBALANCED: "UNBALANCED",
        smcf.BAD_RESULT: "BAD_RESULT",
        smcf.BAD_COST_RANGE: "BAD_COST_RANGE"
    }
    
    if status != smcf.OPTIMAL:
        raise RuntimeError("Flow solver failed.")
    
    # 6) 提取结果
    labels = np.full(n, -1, dtype=int)
    
    for arc_id in range(smcf.NumArcs()):
        if smcf.Flow(arc_id) > 0:
            tail = smcf.Tail(arc_id)
            head = smcf.Head(arc_id)
            
            # 检查是否是点到聚类的边
            if 2 <= tail < 2 + n and 2 + n <= head < 2 + n + k:
                point_idx = tail - 2
                cluster_idx = head - (2 + n)
                labels[point_idx] = cluster_idx
    
    # 验证结果
    if np.any(labels == -1):
        raise RuntimeError("部分点未被分配到聚类")
    
    return labels, centers

def balanced_kmeans_mcmf_fast_v2(points, k, m=3):
    n = len(points)
    if n % k != 0:
        raise ValueError("n must be divisible by k for balanced clustering.")
    cluster_size = n // k

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    initial_labels = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_
    distances = cdist(points, centers)

    smcf = pywrapgraph.SimpleMinCostFlow()

    source = 0
    sink = n + k + 1

    smcf.SetNodeSupply(source, n)
    smcf.SetNodeSupply(sink, -n)

    # source -> points
    for i in range(n):
        smcf.AddArcWithCapacityAndUnitCost(source, i + 1, 1, 0)

    MAX_COST = 1000000

    # points -> clusters
    for i in range(n):
        nearest_clusters = set(np.argsort(distances[i])[:m])
        assigned_cluster = initial_labels[i]
        nearest_clusters.add(assigned_cluster)
        for j in range(k):
            if j in nearest_clusters:
                cost = int(1000 * distances[i, j])
            else:
                cost = MAX_COST
            smcf.AddArcWithCapacityAndUnitCost(i + 1, n + 1 + j, 1, cost)

    # clusters -> sink
    for j in range(k):
        smcf.AddArcWithCapacityAndUnitCost(n + 1 + j, sink, cluster_size, 0)

    status = smcf.Solve()

    if status != smcf.OPTIMAL:
        raise RuntimeError(f"Flow solver failed with status {status}.")

    final_labels = np.full(n, -1, dtype=int)

    # 读取流结果，从点节点流向哪个cluster节点
    for i in range(n):
        arc_count = smcf.NumArcs()
        for arc_id in range(arc_count):
            tail = smcf.Tail(arc_id)
            head = smcf.Head(arc_id)
            flow = smcf.Flow(arc_id)
            if tail == i + 1 and flow > 0 and n + 1 <= head <= n + k:
                final_labels[i] = head - (n + 1)
                break

    return final_labels, centers



def balanced_kmeans_fast(points, k, m=3, max_iters=20):
    n, dim = points.shape
    assert n % k == 0, "n must be divisible by k"
    cluster_size = n // k
    
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(points)
    centers = km.cluster_centers_

    for _ in range(max_iters):
        dists = cdist(points, centers)  # [n, k]
        nearest = np.argsort(dists, axis=1)[:, :m]  # 每个点的最近m个中心
        assigned = np.full(n, -1)
        cluster_load = np.zeros(k, dtype=int)

        # Step: 贪心分配到最近但可用的中心
        for i in range(n):
            for j in nearest[i]:
                if cluster_load[j] < cluster_size:
                    assigned[i] = j
                    cluster_load[j] += 1
                    break
        
        # 如果有未分配点，强制分配（极少数）
        for i in range(n):
            if assigned[i] == -1:
                for j in nearest[i]:
                    assigned[i] = j
                    cluster_load[j] += 1
                    break

        # 更新中心
        new_centers = np.zeros((k, dim))
        counts = np.zeros(k)
        for i in range(n):
            j = assigned[i]
            new_centers[j] += points[i]
            counts[j] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centers[j] /= counts[j]
        centers = new_centers

    return assigned, centers


def compute_dismatrix(points, labels, num_clusters):
    distance_matrix = np.zeros((num_clusters, num_clusters))  # 初始化距离矩阵为无穷大

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            # 获取类 i 和类 j 的所有点
            points_i = points[labels == i]
            points_j = points[labels == j]

            # 计算类 i 和类 j 之间的最小距离
            if len(points_i) > 0 and len(points_j) > 0:
                dist = cdist(points_i, points_j, metric='euclidean')
                min_dist = np.min(dist)
                distance_matrix[i, j] = min_dist
                distance_matrix[j, i] = min_dist  # 距离矩阵是对称的
    return distance_matrix

def extract_effective_tour(tour, target_range=range(50)):
    target_set = set(target_range)
    n = len(tour)

    # Find the start index: first target node with a non-target predecessor.
    start_index = None
    for i in range(n):
        prev = tour[i - 1]  # Wraps around to last element when i = 0
        if tour[i] in target_set and prev not in target_set:
            start_index = i
            break

    if start_index is None:
        # If no transition is found, assume the tour is entirely target nodes.
        return [node for node in tour if node in target_set]

    # Collect target nodes cyclically, avoiding duplicates.
    effective_tour = []
    i = start_index
    first_node = tour[start_index]  # Keep track of the first node
    first_encountered = False  # To prevent duplicate entry at the end
    
    while True:
        if tour[i] in target_set:
            if tour[i] == first_node and first_encountered:
                break  # Stop when we reach the first node again
            effective_tour.append(tour[i])
            first_encountered = True
        i = (i + 1) % n
        if i == start_index:
            break

    return effective_tour

def detect_reversed_tour(tour, target_tour, prev_nodes_length, check_range = 8):
    indices = {node: i for i, node in enumerate(tour)}
    
    # Get first and last node index
    first_node, last_node = target_tour[0], target_tour[-1]
    first_index, last_index = indices[first_node], indices[last_node]

    # Check a few nodes before and after the target tour
    prev_nodes = tour[max(0, first_index - check_range):first_index]
    next_nodes = tour[last_index + 1:min(len(tour), last_index + 1 + check_range)]
    
    # If nodes before `first_node` belong to `next` range, reverse the tour
    target_tour_len =  len(target_tour)
    prev_range = range(target_tour_len, target_tour_len + prev_nodes_length)
    next_range = range(target_tour_len + prev_nodes_length, target_tour_len + 2* prev_nodes_length)
    if all(node not in prev_range for node in prev_nodes) and all(node not in next_range for node in next_nodes):
        target_tour.reverse()

    return target_tour

def circumcenter(A, B, C):
    denominator = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    if np.abs(denominator) < 1e-6:
        return np.mean([A, B, C], axis=0)
    
    def numerator_part(p1, p2, p3):
        return (p1[0]**2 + p1[1]**2) * (p2[1] - p3[1])
    
    Ux = (numerator_part(A, B, C) + numerator_part(B, C, A) + numerator_part(C, A, B)) / denominator
    Uy = (numerator_part(A, C, B) + numerator_part(B, A, C) + numerator_part(C, B, A)) / denominator
    return np.array([Ux, Uy])

def normalize_angle(angle, circle_center):
    angle1 = np.arctan2(angle[1]-circle_center[1], angle[0]-circle_center[0])
    return angle1 % (2*np.pi)


def soft_normalize(global_coords, main_coords):
    min_vals = global_coords.min(axis=0)
    range_vals = np.maximum(global_coords.max(axis=0) - min_vals, 1e-10)
    global_normalized = (global_coords - min_vals) / range_vals

    def sigmoid(x, scale=6, shift=0.5):
        return 1 / (1 + np.exp(-scale * (x - shift)))

    global_scaled = sigmoid(global_normalized) * 0.96 + 0.02
    main_scaled = sigmoid((main_coords - min_vals) / range_vals) * 0.96 + 0.02

    return global_scaled, main_scaled


def hard_normalize(global_coords, main_coords):
    # 计算全局最大范围（保持纵横比的关键）
    min_vals = global_coords.min(axis=0)
    max_vals = global_coords.max(axis=0)
    ranges = max_vals - min_vals
    max_range = max(ranges)  # 按长边比例缩放
    
    # 统一按max_range缩放（保持纵横比）
    global_scaled = (global_coords - min_vals) / max_range
    main_scaled = (main_coords - min_vals) / max_range
    
    # 可选：平移数据到 [0.02, 0.98] 附近
    current_min = global_scaled.min()
    current_max = global_scaled.max()
    target_min, target_max = 0.02, 0.98
    global_scaled = (global_scaled - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    main_scaled = (main_scaled - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    
    return global_scaled, main_scaled

def is_angle_between(angle, start, end):
    # Assumes all angles are normalized to [0, 2π)
    if start <= end:
        return start <= angle <= end
    else:
        return angle >= start or angle <= end


def push_points_relative(points, main_points, factor=2):
    center_points = np.mean(points, axis=0)
    center_main = np.mean(main_points, axis=0)

    offset = (center_points - center_main) * factor
    return points + offset

def generate_relative_points(prev_points: np.ndarray, main_points: np.ndarray, redundancy_length: int) -> np.ndarray:
    # 1. 计算两个子图的中心
    center_prev = np.mean(prev_points, axis=0)
    center_main = np.mean(main_points, axis=0)

    # 2. 计算从主图中心到前图中心的单位方向向量
    direction = center_prev - center_main
    norm = np.linalg.norm(direction)
    # if norm == 0: raise ValueError("prev_points 和 main_points 的中心重合，无法定义方向")
    unit_dir = direction / norm

    rel = main_points - center_main  # 每个点相对主中心的向量
    proj_lengths = rel.dot(unit_dir)  # 在 direction 方向上的投影长度
    max_len = proj_lengths.max()
    boundary_pt = center_main + unit_dir * max_len * 1.2 # 映射点：中心加上最大投影长度方向偏移

    relative_points = np.linspace(boundary_pt, center_prev, redundancy_length) #均匀放置冗余点

    return relative_points, center_prev


def chose_closest_points(points, main_points, topk=5):
    target_mean = np.mean(main_points, axis=0)
    distances = np.linalg.norm(points - target_mean, axis=1)
    topk_indices = np.argpartition(distances, topk)[:topk]
    topk_points = points[topk_indices]

    return topk_points

def generate_bridge_points_relative_cycle(main_points, left_points, right_points, num=8, noise_x=5, noise_y=3):
    # Compute centers
    center_main, center_left, center_right = np.array([np.mean(points, axis=0) for points in [main_points, left_points, right_points]])
    
    # Compute circumcenter and radius from the three centers
    circle_center = circumcenter(center_main, center_left, center_right)
    radius = np.linalg.norm(center_left - circle_center)
    
    # Compute angles of the three centers relative to circle_center
    angle_main, angle_left, angle_right = [normalize_angle(c, circle_center) for c in [center_main, center_left, center_right]]
    
    # Choose the suitable arc based on the main cluster angle
    arc_length = (angle_right - angle_left) % (2*np.pi)
    use_reverse_arc = is_angle_between(angle_main, angle_left, angle_right) or arc_length > np.pi
    start_angle, end_angle = (angle_right, angle_left + 2*np.pi) if use_reverse_arc else (angle_left, angle_right)

    # Generate evenly spaced angles along the chosen arc
    angles = np.linspace(start_angle, end_angle, num)
    bridge = np.column_stack((
        circle_center[0] + radius * np.cos(angles),
        circle_center[1] + radius * np.sin(angles) ))
    bridge_points = bridge + np.random.uniform([-noise_x, -noise_y], [noise_x, noise_y], (num, 2))
    
    return bridge_points


def generate_bridge_points_relative_square(main_points, left_points, right_points, num=8, noise_x=5, noise_y=3):
    # 计算三个簇的中心
    A = np.mean(main_points, axis=0)
    B = np.mean(left_points, axis=0)
    C = np.mean(right_points, axis=0)

    vec_AB = B - A
    vec_AC = C - A
    vec_BC = C - B
    if np.dot(vec_AB, vec_AC) < 0:  # 角BAC > 90度
        M = (B + C) / 2  # BC的中点
        directioner = np.array([vec_BC[1], -vec_BC[0]])
        if np.dot(M - A, directioner) > 0:
            directioner = -directioner
        D = M - directioner / np.linalg.norm(directioner) * np.linalg.norm(B - C) / 2  # D点位置
    else:
        D = B + C - A  # 平行四边形规则

    # 找到left_points和right_points中距离D最近的点
    left_start = left_points[np.argmin(np.linalg.norm(left_points - D, axis=1))]
    right_start = right_points[np.argmin(np.linalg.norm(right_points - D, axis=1))]

    # 在BD和CD线上生成点（从外侧向D点采样）
    k = num // 2
    bd_points = np.linspace(left_start, D, k + 2)[1:-1]  # 从left_start到D的中间点
    cd_points = np.linspace(right_start, D, k + 2)[1:-1]  # 从right_start到D的中间点

    # 合并点并添加噪声
    bridge_points = np.vstack((bd_points, cd_points))
    bridge_points += np.random.uniform([-noise_x, -noise_y], [noise_x, noise_y], bridge_points.shape)

    return bridge_points


def generate_bridge_points_relative_fused(main_points, left_point, right_point, num=8, noise_x=5, noise_y=3):
    A = np.mean(main_points, axis=0)
    B = left_point
    C = right_point

    vec_AB = B - A
    vec_AC = C - A
    vec_BC = C - B

    dot_product = np.dot(vec_AB, vec_AC)
    cos_theta = dot_product / (np.linalg.norm(vec_AB) * np.linalg.norm(vec_AC))

    if cos_theta > -0.7071:  # 角BAC < 135度
        D = B + C - A  # 平行四边形规则
        # 找到left_points和right_points中距离D最近的点
        left_start = left_point
        right_start = right_point

        # 在BD和CD线上生成点（从外侧向D点采样）
        k = num // 2
        bd_points = np.linspace(left_start, D, k + 2)[1:-1]  # 从left_start到D的中间点
        cd_points = np.linspace(right_start, D, k + 2)[1:-1]  # 从right_start到D的中间点

        # 合并点并添加噪声
        bridge_points = np.vstack((bd_points, cd_points))
        bridge_points += np.random.uniform([-noise_x, -noise_y], [noise_x, noise_y], bridge_points.shape)                       

    else:
        center = A
        radius = np.linalg.norm(B - A)

        def angle_of(P): return np.arctan2(P[1] - center[1], P[0] - center[0])

        theta_B = angle_of(B)
        theta_C = angle_of(C)
        if theta_C < theta_B : theta_C += 2 * np.pi

        dtheta = (theta_C - theta_B) % (2 * np.pi)

        if dtheta <= np.pi:
            angles = np.linspace(theta_B, theta_C, num)
        else:
            # 逆向小于 π 的弧
            angles = np.linspace(theta_B, theta_C - 2*np.pi, num)
        # angles = np.linspace(theta_B, theta_C, num)

        # 生成弧上的点
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        bridge_points = np.stack([xs, ys], axis=1)
                         
    return bridge_points

def convert_points(points):
    return [list(map(float, p)) for p in points]

def compute_tour_cost(merged_points, global_tour_indices):
    tour_coords = merged_points[global_tour_indices]
    total_cost = np.sum(np.linalg.norm(tour_coords - np.roll(tour_coords, shift=-1, axis=0), axis=1))
    return total_cost

def compute_path_cost(merged_points, global_tour_indices):
    path_coords = merged_points[global_tour_indices]
    total_cost = np.sum(np.linalg.norm(path_coords[1:] - path_coords[:-1], axis=1))
    return total_cost

def two_opt_path(merged_points, path, improvement_threshold=0.001, max_iter=30):
    best_path = path.copy()
    best_cost = compute_path_cost(merged_points, best_path)
    n = len(best_path)

    for _ in range(max_iter):
        improvement_found = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                candidate = np.concatenate((best_path[:i], best_path[i:j+1][::-1], best_path[j+1:]))
                candidate_cost = compute_path_cost(merged_points, candidate)
                if best_cost - candidate_cost > improvement_threshold * best_cost:
                    best_path, best_cost = candidate, candidate_cost
                    improvement_found = True
                    break
                
            if improvement_found: break
        if not improvement_found: break
    return best_path, best_cost

def get_top_long_edges(merged_points, global_tour, top_percentage=2):
    tour_coords = merged_points[global_tour]
    edge_lengths = np.linalg.norm(tour_coords - np.roll(tour_coords, -1, axis=0), axis=1)

    threshold_idx = int(len(edge_lengths) * (1 - top_percentage / 100))
    top_edges_idx = np.argsort(edge_lengths)[threshold_idx:]
    return top_edges_idx


def targeted_2opt_long_edges(merged_points, global_tour, window_size=50, improvement_threshold=0.001, max_iter=100, top_percentage = 2):
    """Optimize the tour by applying 2-opt on the top k longest edges."""
    top_edges_idx = get_top_long_edges(merged_points, global_tour, top_percentage)
    
    for max_idx in top_edges_idx:
        start_idx, end_idx = max(0, max_idx - window_size // 2), min(len(global_tour), max_idx + window_size // 2 + 2)
        window = global_tour[start_idx:end_idx]

        # Optimize the sub-tour
        new_window, _ = two_opt_path(merged_points, window, improvement_threshold, max_iter)
        global_tour[start_idx:end_idx] = new_window

    return global_tour, compute_tour_cost(merged_points, global_tour), 0

def randomized_2opt(merged_points, global_tour, num_iterations=200, window_size=50, improvement_threshold=0.001, max_iter=100):
    n = len(global_tour)
    
    with tqdm(range(num_iterations), desc="Randomized edges 2-opt") as pbar:
        for _ in range(num_iterations):
            start_idx  = np.random.randint(0, n - window_size)
            end_idx = start_idx + window_size
            window = global_tour[start_idx:end_idx]

            new_window, _ = two_opt_path(merged_points, window, improvement_threshold, max_iter)
            global_tour[start_idx:end_idx] = new_window
            pbar.update()

    return global_tour, compute_tour_cost(merged_points, global_tour), pbar.format_dict['elapsed']

def process_subgraph(main_points, main_cluster_id, prev_points, next_points, cluster_global_indices, redundancy_length = 16, if_plot_subgraph=False):
    chosen_prev_points, chosen_next_points = prev_points[:redundancy_length], next_points[:redundancy_length]
    # chosen_prev_points = chose_closest_points(prev_points, main_points, topk=redundancy_length)
    # chosen_next_points = chose_closest_points(next_points, main_points, topk=redundancy_length)
    prev_shifted = push_points_relative(chosen_prev_points, main_points)
    next_shifted = push_points_relative(chosen_next_points, main_points)
    bridge_points = generate_bridge_points_relative_square(main_points, prev_shifted, next_shifted, num=8)

    enhanced_points = np.concatenate([main_points, prev_shifted,  next_shifted, bridge_points])

    try:
        coord_dict = build_coord_dict(enhanced_points, prefix=str(main_cluster_id))
        solver = elkai.Coordinates2D(coord_dict)
        tour_nodes = solver.solve_tsp()
        tour = [int(node.split('_')[-1]) for node in tour_nodes]
    except Exception as e:
        print(f"子图{main_cluster_id}求解失败: {e}")
        return None, None

    if if_plot_subgraph: plot_cluster_tour(enhanced_points, main_points, tour, main_cluster_id)

    # Trans cluster index to global index
    main_indices = extract_effective_tour(tour[:-1], target_range=range(len(main_points)))
    # Check if the tour is reversed, proving in and out node
    main_indices = detect_reversed_tour(tour, main_indices, redundancy_length)
    main_global_indices = cluster_global_indices[main_indices]

    # print(main_indices)
    if len(main_indices) != len(main_points):  print(main_cluster_id,len(main_indices))
    return main_global_indices, enhanced_points[tour]

def main_oldLKH(args):
    setup_logging(args, args.logfile_path ,logging.INFO if not args.debug else logging.DEBUG)

    dataset = TSPGraphDataset(args.data_path, sparse_factor=-1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    merged_points = []
    for batch_idx, (_, points, _, _) in enumerate(dataloader):
        if batch_idx >= args.used_num: break
        merged_points.append(points.squeeze(0).numpy() * 10000)
    merged_points = np.concatenate(merged_points, axis=0)

    # Cluster and Adjust Labels for Diffusion Model
    adjusted_labels, centers = balanced_kmeans_mcmf(merged_points, args.cluster_num)
    if args.if_plot_clusters : plot_clusters(merged_points, adjusted_labels, centers)

    print("Balanced Kmeans finished")

    # Solve TSP for Hyper Huge Graph with Clusters
    centers = [merged_points[adjusted_labels==i].mean(axis=0) for i in range(args.cluster_num)]
    super_coords = build_coord_dict(centers, prefix="super")
    solver = elkai.Coordinates2D(super_coords)
    super_tour_nodes = solver.solve_tsp()
    super_tour = [int(node.split('_')[-1]) for node in super_tour_nodes[:-1]]
    # distance_matrix = compute_dismatrix(merged_points, adjusted_labels, args.cluster_num)
    # distance_matrix_int = (distance_matrix).astype(int)  # 缩放并取整
    # super_tour = elkai.solve_int_matrix(distance_matrix_int)
    
    global_tour_indices = []
    full_path = []

    with tqdm(super_tour) as pbar:
        for idx in super_tour:
            pos = super_tour.index(idx) 
            prev_idx = super_tour[pos-1] if pos > 0 else super_tour[-2]
            next_idx = super_tour[pos+1] if pos < len(super_tour)-1 else super_tour[0]
            
            # Get the target cluster ordinaries and indices
            cluster_global_indices = np.where(adjusted_labels == idx)[0]
            current_cluster_points = merged_points[cluster_global_indices]
            
            # Get the context cluster ordinaries and indices
            prev_points = merged_points[adjusted_labels == prev_idx]
            next_points = merged_points[adjusted_labels == next_idx]
            
            # Solve Cluster TSP and get the global indices
            segment_indices, sub_path = process_subgraph(
                current_cluster_points,  idx, prev_points,  next_points, cluster_global_indices, redundancy_length=args.redundancy_length, if_plot_subgraph = args.if_plot_subgraph)
            
            
            if segment_indices is not None:
                global_tour_indices.extend(segment_indices)
                full_path.append(sub_path)
            pbar.update(1)
        
    
    total_cost = compute_tour_cost(merged_points, np.array(global_tour_indices))
    print(f"Clustered Subgraph with LKH, tour cost: {total_cost:.2f}, using time: {pbar.format_dict['elapsed']:.2f}")
    logging.info(f"Clustered Subgraph with LKH, tour cost: {total_cost:.2f}, using time: {pbar.format_dict['elapsed']:.2f}")

    # new_tour, new_cost, used_time = targeted_2opt_long_edges(merged_points, np.array(global_tour_indices), top_percentage= args.long_2opt_percent, window_size= args.long_2opt_window)
    # print(f"After optimizing long edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")
    # logging.info(f"After optimizing long edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")
    # new_tour, new_cost, used_time = randomized_2opt(merged_points, new_tour, num_iterations= args.rand_2opt_number, window_size= args.rand_2opt_window)
    # print(f"After optimizing random edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")
    # logging.info(f"After optimizing random edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")


    starttime = time.time()
    cpp_points = convert_points(merged_points)
    optimized_tour = tsp_optimizer.longedges_2opt(cpp_points, global_tour_indices, args.long_2opt_window, 0.001, 30, args.long_2opt_percent)
    new_cost = tsp_optimizer.compute_path_cost(cpp_points, optimized_tour)
    used_time = time.time()-starttime
    print(f"After optimizing long edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")
    logging.info(f"After optimizing long edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")

    starttime = time.time()
    optimized_tour = tsp_optimizer.randomized_2opt(cpp_points, optimized_tour, args.rand_2opt_number, args.rand_2opt_window, 0.001, 30)
    new_cost = tsp_optimizer.compute_path_cost(cpp_points, optimized_tour)
    used_time = time.time()-starttime
    print(f"After optimizing random edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")
    logging.info(f"After optimizing random edges, tour cost: {new_cost:.2f}, using time: {used_time:.2f}")
    
    
    if args.if_plot_global:
        plot_global(merged_points, adjusted_labels, np.array(optimized_tour), centers, super_tour)



# def process_test_end_heatmap(xt, solution_points, sequential_sampling, cluster_global_indices, main_points, parallel_sampling = 1, test_2opt_iterations =1000, sparser = False):
#     np_edge_index = None
#     small_adjs, all_points,offset, cluster_map  = [], [], 0, []

#     if args.diffusion_type == 'gaussian':
#         adj_mats = xt.cpu().detach().numpy() * 0.5 + 0.5
#     else:
#         adj_mats = xt.float().cpu().detach().numpy() + 1e-6
#     subgraph_num = len(adj_mats[0])

#     for ss in range(sequential_sampling):
#         assert ss == 0, "SS > 1, Needs further judgement"
#         for idx in range(subgraph_num):
#             # 裁剪前 50×50
#             small_adj = adj_mats[ss][idx][:50, :50]
#             small_adjs.append(small_adj)
#             pts = main_points[idx]   
#             all_points.append(pts)

#             cluster_map.append((offset, offset + 50))
#             offset += 50

#         # 2. 拼成 10000×10000 的大邻接矩阵
#         big_size = subgraph_num * 50
#         big_adj  = np.zeros((big_size, big_size), dtype=small_adjs[0].dtype)
#         for g, mat in enumerate(small_adjs):
#             s = g * 50
#             big_adj[s:s+50, s:s+50] = mat

#         big_points = np.vstack(all_points)

#         # 4. 一次性调用 merge_tours + 2-opt
#         #    注意：merge_tours 接受的 adj_mat 需要形状 (1, big_size, big_size)
#         tours, merge_iterations = merge_tours(
#             adj_mat       = big_adj[None, :, :],
#             np_points = big_points,
#             edge_index_np = np_edge_index,
#             sparse_graph  = sparser,
#             parallel_sampling = parallel_sampling,
#         )
#         print("Mergement finish")

#         solved_tour, ns = batched_two_opt_torch(
#             big_points.astype("float64"),
#             np.array(tours).astype("int64"),
#             max_iterations = test_2opt_iterations,
#             device= 'cuda:0'
#         )
#         print("batched_two_opt_torch finish")

#         final_tour = solved_tour[0].tolist()

#         mapping = np.concatenate(cluster_global_indices, axis=0)
#         global_tour = mapping[final_tour]  

#     return global_tour


def xt2large_noise(groups, xt_grouped, scale2_info, intra_noise_prop, inter_noise_prop):
    [scale2_times, scale2_main, scale2_extend] = scale2_info
    xt_groups = []
    for g in range(groups):
        # 先构造 5 个子图的 block-diagonal (250×250)
        block_diag = torch.block_diag(*[xt_grouped[g, k] for k in range(scale2_times)])
        # 创建 280×280 的 padded 矩阵，并将子图数据填入左上角
        padded = torch.zeros(scale2_extend, scale2_extend, device=block_diag.device)
        padded[:scale2_main, :scale2_main] = block_diag


        # —— 一、在三个簇内部加入噪声 —— 
        # 簇 1: [250:262], 簇 2: [262:268], 簇 3: [268:280]
        cluster_starts = [scale2_main, scale2_main + 8, scale2_main + 16]  # [250, 262, 268]
        cluster_ends   = [scale2_main + 8, scale2_main + 16, scale2_main + 30]  # [262, 268, 280]
        for start, end in zip(cluster_starts, cluster_ends):
            size = end - start  # 簇内部大小
            # 生成 [size × size] 的随机噪声矩阵（对角和对称均可），这里只直接用全噪声方块
            noise_block = (torch.rand(size, size, device=block_diag.device) < intra_noise_prop).float()
            padded[start:end, start:end] = noise_block

        # —— 二、在每对相邻区块（子图↔子图、子图↔簇、簇↔簇）之间加入噪声 —— 
        # 定义所有“块”的边界索引列表：
        #   子图 1 ~ 5: [0:50],[50:100],[100:150],[150:200],[200:250]
        #   簇 1~3: [250:262],[262:268],[268:280]
        temp_scale = scale2_main//5
        boundaries = [0, temp_scale*1, temp_scale*2, temp_scale*3, temp_scale*4, temp_scale*5, temp_scale*5+8, temp_scale*5+16, temp_scale*5+30]
        # 遍历每对相邻区块 i 和 i+1
        for i in range(len(boundaries) - 2):
            a_start, a_end = boundaries[i],   boundaries[i+1]
            b_start, b_end = boundaries[i+1], boundaries[i+2]
            size_a = a_end - a_start
            size_b = b_end - b_start

            # 生成 [size_a × size_b] 的随机噪声
            inter_noise = (torch.rand(size_a, size_b, device=block_diag.device) < inter_noise_prop).float()
            # 将噪声填到 [a_start:a_end, b_start:b_end] 和对称位置 [b_start:b_end, a_start:a_end]
            padded[a_start:a_end, b_start:b_end] = inter_noise
            padded[b_start:b_end, a_start:a_end] = inter_noise.t()

        xt_groups.append(padded)


    return xt_groups





def process_cluster(idx, super_tour, adjusted_labels, merged_points, redundancy_length, bridge_length):
    """ 处理单个 idx 的点 """
    pos = super_tour.index(idx)
    prev_idx = super_tour[pos - 1] if pos > 0 else super_tour[-1]
    next_idx = super_tour[pos + 1] if pos < len(super_tour) - 1 else super_tour[0]

    cluster_global_indices = np.where(adjusted_labels == idx)[0]
    main_points = merged_points[cluster_global_indices]

    prev_points = merged_points[adjusted_labels == prev_idx][:redundancy_length]
    next_points = merged_points[adjusted_labels == next_idx][:redundancy_length]

    prev_shifted = push_points_relative(prev_points, main_points, factor= 0.5)
    next_shifted = push_points_relative(next_points, main_points, factor= 0.5)
    bridge_points = generate_bridge_points_relative_square(main_points, prev_shifted, next_shifted, num=bridge_length)

    enhanced_points = np.concatenate([main_points, prev_shifted, next_shifted, bridge_points])
    enhanced_points, main_points =  hard_normalize(enhanced_points, main_points)

    # plot_cluster_wotour(enhanced_points, main_points, idx)
    
    return idx, enhanced_points, cluster_global_indices, main_points

def process_cluster2(super_tour, cluster_global_indices1, merged_points, redundancy_length, bridge_length, scale1_subcount, scale2_times):
    cluster_global_indices1
    enhanced_points2, cluster_global_indices2, main_points2_t = [],[],[]
    for i in range(0, scale1_subcount, scale2_times):  
        chunkc = cluster_global_indices1[i:i+scale2_times]
        merged_chunkc = np.array(chunkc).reshape(-1).tolist()
        cluster_global_indices2.append(np.array(merged_chunkc))

        # chunkp = main_points1[i:i+4]
        # merged_chunkp = np.array(chunkp).reshape(-1, 2)
        # main_points2.append(merged_chunkp)
        main_points2_t.append(merged_points[merged_chunkc])

    main_points2 = []
    scale2_subcount = int(scale1_subcount/scale2_times)

    for idx in range(scale2_subcount):
        prev_idx = idx-1 if idx > 0 else scale2_subcount-1
        next_idx = idx+1 if idx < scale2_subcount-1 else 0

        main_points = main_points2_t[idx]

        prev_points = main_points2_t[prev_idx][:redundancy_length]
        next_points = main_points2_t[next_idx][:redundancy_length]

        prev_shifted = push_points_relative(prev_points, main_points, factor= 0.5)
        next_shifted = push_points_relative(next_points, main_points, factor= 0.5)
        bridge_points = generate_bridge_points_relative_square(main_points, prev_shifted, next_shifted, num=bridge_length)

        enhanced_points = np.concatenate([main_points, prev_shifted, bridge_points, next_shifted])
        enhanced_points, main_points =  hard_normalize(enhanced_points, main_points)
        enhanced_points2.append(enhanced_points)
        main_points2.append(main_points)
    
    return enhanced_points2, cluster_global_indices2, main_points2

def process_cluster3(idx, super_tour, adjusted_labels, merged_points, redundancy_length, bridge_length):
    """ 处理单个 idx 的点 """
    pos = super_tour.index(idx)
    prev_idx = super_tour[pos - 1] if pos > 0 else super_tour[-1]
    next_idx = super_tour[pos + 1] if pos < len(super_tour) - 1 else super_tour[0]

    cluster_global_indices = np.where(adjusted_labels == idx)[0]
    main_points = merged_points[cluster_global_indices]

    prev_points = merged_points[adjusted_labels == prev_idx]
    next_points = merged_points[adjusted_labels == next_idx]

    prev_shifted, prev_center = generate_relative_points(prev_points, main_points, redundancy_length)
    next_shifted, next_center = generate_relative_points(next_points, main_points, redundancy_length)
    bridge_points = generate_bridge_points_relative_fused(main_points, prev_center, next_center, num=bridge_length)

    enhanced_points = np.concatenate([main_points, prev_shifted, next_shifted, bridge_points])
    enhanced_points, main_points =  hard_normalize(enhanced_points, main_points)

    # plot_cluster_wotour(enhanced_points, main_points, idx)
    
    return idx, enhanced_points, cluster_global_indices, main_points

def process_clusters_wtour(super_tour, adjusted_labels, merged_points, redundancy_length, bridge_length):
    """一次性处理 super_tour 上的所有 idx，返回 enhanced_data, cluster_global_indices, main_points"""
    enhanced_data = []
    cluster_global_indices = []
    main_points_list = []

    for idx in super_tour:
        # 复用你原来的单点处理逻辑
        pos = super_tour.index(idx)
        prev_idx = super_tour[pos - 1] if pos > 0 else super_tour[-1]
        next_idx = super_tour[pos + 1] if pos < len(super_tour) - 1 else super_tour[0]

        cluster_global_index = np.where(adjusted_labels == idx)[0]
        main_points = merged_points[cluster_global_index]

        prev_points = merged_points[adjusted_labels == prev_idx]
        next_points = merged_points[adjusted_labels == next_idx]

        prev_shifted, prev_center = generate_relative_points(prev_points, main_points, redundancy_length)
        next_shifted, next_center = generate_relative_points(next_points, main_points, redundancy_length)
        bridge_points = generate_bridge_points_relative_fused(main_points, prev_center, next_center, num=bridge_length)

        enhanced_points = np.concatenate([main_points, prev_shifted, next_shifted, bridge_points])
        enhanced_points, main_points =  hard_normalize(enhanced_points, main_points)

        enhanced_data.append(enhanced_points)
        cluster_global_indices.append(cluster_global_index)
        main_points_list.append(main_points)

    return enhanced_data, cluster_global_indices, main_points_list

def process_clusters_wotour(super_tour, adjusted_labels, merged_points):
    """一次性处理 super_tour 上的所有 idx，返回 enhanced_data, cluster_global_indices, main_points"""
    enhanced_data = []
    cluster_global_indices = []
    main_points_list = []

    idx = 0
    cluster_global_index = np.where(adjusted_labels == idx)[0]
    main_points = merged_points[cluster_global_index]

   
    _, main_points =  hard_normalize(main_points, main_points)

    cluster_global_indices.append(cluster_global_index)
    main_points_list.append(main_points)

    return main_points_list, cluster_global_indices, main_points_list


def process_clusters_wtour_v2(super_tour, adjusted_labels, merged_points, redundancy_length, bridge_length):
    """带强化边并写入"""
    enhanced_data = []
    cluster_global_indices = []
    main_points_list = []

    import random 
    for idx in super_tour:
        # 复用你原来的单点处理逻辑
        pos = super_tour.index(idx)
        prev_idx = super_tour[pos - 1] if pos > 0 else super_tour[-1]
        next_idx = super_tour[pos + 1] if pos < len(super_tour) - 1 else super_tour[0]

        cluster_global_index = np.where(adjusted_labels == idx)[0]
        main_points = merged_points[cluster_global_index]

        prev_points = merged_points[adjusted_labels == prev_idx]
        next_points = merged_points[adjusted_labels == next_idx]

        prev_shifted, prev_center = generate_relative_points(prev_points, main_points, redundancy_length)
        next_shifted, next_center = generate_relative_points(next_points, main_points, redundancy_length)
        bridge_points = generate_bridge_points_relative_fused(main_points, prev_center, next_center, num=bridge_length)

        enhanced_points = np.concatenate([main_points, prev_shifted, next_shifted, bridge_points])
        enhanced_points, main_points =  hard_normalize(enhanced_points, main_points)

        # gt_path = list(range(1, len(enhanced_points) + 1))
        # random.shuffle(gt_path)
        # gt_path.append(gt_path[0])  # 使路径闭合
        
        # # 格式化数据
        # coord_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in enhanced_points])
        # gt_str = ' '.join(map(str, gt_path))
        # data_line = f"{coord_str} output {gt_str}\n"
        
        # # 写入文件（追加模式）
        # with open("tsp_data.txt", 'a') as f:  # 'a'表示追加模式
        #     f.write(data_line)

        enhanced_data.append(enhanced_points)
        cluster_global_indices.append(cluster_global_index)
        main_points_list.append(main_points)

    return enhanced_data, cluster_global_indices, main_points_list

def process_test_end(xt, solution_points, sequential_sampling, cluster_global_indices, main_points, parallel_sampling = 1, test_2opt_iterations =1000, sparser = False):
    np_edge_index = None
    stacked_tours = []
    if args.diffusion_type == 'gaussian':
        adj_mats = xt.cpu().detach().numpy() * 0.5 + 0.5
    else:
        adj_mats = xt.float().cpu().detach().numpy() + 1e-6
    subgraph_num = len(adj_mats[0])

    for ss in range(sequential_sampling):
        assert ss == 0, "SS > 1, Needs further judgement"
        difusco_solved_tours, ss_stacked_tours = [], []
        d_len, l_len = 0, 0
        with tqdm(total=subgraph_num, desc='Solving DIFUSCO Tour') as pbar:
            for idx in range(subgraph_num):
                adj_mat = np.expand_dims(adj_mats[ss][idx], axis=0)
                adj_mat_4view = adj_mat[0]
                merged_points = solution_points[ss][idx]
                tours, merge_iterations = merge_tours_v2(
                    adj_mat, merged_points, np_edge_index,
                    sparse_graph=sparser,
                    parallel_sampling=parallel_sampling,
                )

                # solved_tour, ns = batched_two_opt_torch( merged_points.astype("float64"), np.array(tours).astype('int64'), max_iterations=test_2opt_iterations)
                solved_tour, _ = two_opt_numpy( merged_points.astype("float64"), np.array(tours[0][:-1]).astype('int64'), max_iterations=test_2opt_iterations)
                solved_tour = [solved_tour]
                # difusco_solved_tours.append(solved_tour[0].tolist())
                difusco_solved_tours.append(solved_tour[0].tolist())
                pbar.update()

                plot_cluster_tour(merged_points, main_points[idx], solved_tour[0].tolist(), idx, './figures/tsp_250Diff/')

                def tour_length(pts, tour):
                    return sum(np.linalg.norm(pts[tour[i]] - pts[tour[(i+1)%len(tour)]]) for i in range(len(tour)))
                
                # coord_dict = build_coord_dict(merged_points*1000, prefix=str(idx))
                # solver = elkai.Coordinates2D(coord_dict)
                # tour_nodes = solver.solve_tsp()
                # LKH_tour = [int(node.split('_')[-1]) for node in tour_nodes]
                # plot_cluster_tour(merged_points, main_points[idx], LKH_tour, idx, './figures/tsp_250LKH/')
                # l_len += tour_length(merged_points, LKH_tour)
                
                d_len += tour_length(merged_points, solved_tour[0].tolist())
            print(f"diffusion len={d_len:.3f},  LKH len={l_len:.3f}")
        
        with tqdm(total=subgraph_num, desc='Extracting Subgraph Tour') as pbar:
            difusco_solved_tours = np.array(difusco_solved_tours)
            for idx in range(subgraph_num):
                solved_tour, cluster_global_index, main_point = difusco_solved_tours[idx], cluster_global_indices[idx], main_points[idx]
                main_index = extract_effective_tour(solved_tour[:-1], target_range=range(len(main_point)))
                # Check if the tour is reversed, proving in and out node
                main_index = detect_reversed_tour(solved_tour, main_index, 8, 10)
                solved_tours = cluster_global_index[main_index]


                # if len(main_index) != 250:  print(idx, "Error",len(main_index))
                pbar.update()

                ss_stacked_tours.extend(solved_tours)  

    global_tour_indices = ss_stacked_tours #Temporarily with ss=1

    return global_tour_indices

def process_test_end_2x(xt, solution_points, sequential_sampling, cluster_global_indices, main_points, parallel_sampling = 1, test_2opt_iterations =1000, sparser = False):
    np_edge_index = None
    stacked_tours = []
    if args.diffusion_type == 'gaussian':
        adj_mats = xt.cpu().detach().numpy() * 0.5 + 0.5
    else:
        adj_mats = xt.float().cpu().detach().numpy() + 1e-6
    subgraph_num = len(adj_mats[0])

    difusco_solved_tours, ss_stacked_tours = [], []
    d_len, l_len = 0, 0
    with tqdm(total=subgraph_num, desc='Solving DIFUSCO Tour') as pbar:
        for idx in range(subgraph_num):
            adj_mat = np.stack([adj_mats[ss][idx] for ss in range(sequential_sampling)], axis=0)
            adj_mat_4view = adj_mat[0]
            merged_points = solution_points[0][idx]
            tours, merge_iterations = merge_tours_MCTS_2x(
                adj_mat, merged_points, np_edge_index,
                sparse_graph=sparser,
                parallel_sampling=sequential_sampling,
            )

            solved_tour, ns = batched_two_opt_torch(
                merged_points.astype("float64"), np.array(tours).astype('int64'),
                max_iterations=test_2opt_iterations)
            difusco_solved_tours.append(solved_tour[0].tolist())
            pbar.update()

            # plot_cluster_tour(merged_points, main_points[idx], solved_tour[0].tolist(), idx, './figures/tsp_250Diff/')

            def tour_length(pts, tour):
                return sum(np.linalg.norm(pts[tour[i]] - pts[tour[(i+1)%len(tour)]]) for i in range(len(tour)))
            
            d_len += tour_length(merged_points, solved_tour[0].tolist())
        print(f"diffusion len={d_len:.3f},  LKH len={l_len:.3f}")
        
        with tqdm(total=subgraph_num, desc='Extracting Subgraph Tour') as pbar:
            difusco_solved_tours = np.array(difusco_solved_tours)
            for idx in range(subgraph_num):
                solved_tour, cluster_global_index, main_point = difusco_solved_tours[idx], cluster_global_indices[idx], main_points[idx]
                main_index = extract_effective_tour(solved_tour[:-1], target_range=range(len(main_point)))
                # Check if the tour is reversed, proving in and out node
                main_index = detect_reversed_tour(solved_tour, main_index, args.redundancy_length, args.redundancy_length+1)
                solved_tours = cluster_global_index[main_index]


                # if len(main_index) != 250:  print(idx, "Error",len(main_index))
                pbar.update()

                ss_stacked_tours.extend(solved_tours)  

    global_tour_indices = ss_stacked_tours #Temporarily with ss=1

    return global_tour_indices







from typing import List, Optional, Tuple
Coordinate = Tuple[float, float]
Path = List[int]

def euclidean_distance(a: Coordinate, b: Coordinate) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)

def path_length(path: Path, coords: List[Coordinate]) -> float:
    return sum(euclidean_distance(coords[path[i]], coords[path[(i+1) % len(path)]])
               for i in range(len(path)))

class SlidingWindowGreedyOptimizer:
    def __init__(self, coords: List[Coordinate]):
        self.coords = coords
        self.n = len(coords)

    def greedy_tour(self) -> Path:
        # 最近邻启发式
        unvisited = set(range(self.n))
        current = 0
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            next_node = min(unvisited, key=lambda x: euclidean_distance(self.coords[current], self.coords[x]))
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour

    def _two_opt_segment(self, segment: Path) -> Path:
        # 2-opt on segment with fixed endpoints
        best = segment.copy()
        best_len = path_length(best, self.coords)
        m = len(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, m - 2):
                for j in range(i + 1, m - 1):
                    if j - i == 1:
                        continue
                    cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    l = path_length(cand, self.coords)
                    if l < best_len:
                        best, best_len = cand, l
                        improved = True
        return best

    def _three_opt_segment(self, segment: Path) -> Path:
        # 简化的3-opt，只试几种重连
        best = segment.copy()
        best_len = path_length(best, self.coords)
        m = len(best)
        for i in range(1, m - 4):
            for j in range(i + 2, m - 2):
                for k in range(j + 2, m):
                    # 三段 [0:i],[i:j],[j:k]
                    A, B, C = best[:i], best[i:j], best[j:k]
                    D = best[k:]
                    # 四种case
                    cases = [A + B[::-1] + C + D,
                             A + B + C[::-1] + D,
                             A + C + B + D,
                             A + C[::-1] + B[::-1] + D]
                    for cand in cases:
                        if len(cand) != m or len(set(cand)) != m:
                            continue
                        l = path_length(cand, self.coords)
                        if l < best_len:
                            best, best_len = cand, l
                            # 继续外层循环
        return best

    def optimize(self, initial: Path = None,
                 window_size: int = 250,
                 step: int = 50,
                 two_opt: bool = True,
                 three_opt: bool = True,
                 max_iter: int = 3) -> Path:
        # 初始路径
        path = initial or self.greedy_tour()
        n = self.n
        window_size = min(window_size, n)
        step = max(1, step)
        improved = True
        passes = 0
        
        while improved and passes < max_iter:
            improved = False
            passes += 1
            
            with tqdm(total=int(n//step), desc='Optimizing DIFUSCO Tour') as pbar:
                for start in range(0, n, step):
                    # 窗口区间，保留端点
                    end = (start + window_size) % n
                    if start < end:
                        idx = list(range(start, end+1))
                    else:
                        idx = list(range(start, n)) + list(range(0, end+1))
                    # 提取 segment
                    segment = [path[i] for i in idx]
                    # 优化
                    if two_opt:
                        segment = self._two_opt_segment(segment)
                    if three_opt:
                        segment = self._three_opt_segment(segment)
                    # 回写
                    new_path = path.copy()
                    for i, node in zip(idx, segment):
                        new_path[i] = node
                    # 验证并接受
                    if path_length(new_path, self.coords) + 1e-8 < path_length(path, self.coords):
                        path = new_path
                        improved = True
        # 最终检查
        assert sorted(path) == list(range(n)), "路径缺失或重复节点"
        return path
