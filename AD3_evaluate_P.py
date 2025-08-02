# Asyncio
import os
import elkai
import torch
import numpy as np
import argparse
import logging, time
from tqdm import tqdm as tqdm_orig
import torch.multiprocessing as mp
import threading
import copy
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from argparse import Namespace
import random
import ast

from utils.tsp_red_utils import *
from utils.MCTS_utils import *
from co_datasets.tsp_graph_dataset_old import TSPGraphDataset
from torch.utils.data import DataLoader, Dataset
from co_datasets.tsp_graph_dataset_old import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from pl_tsp_model import TSPModel_v2 as TSPModel
from pl_tsp_model import TSPDiffusionDataset_v3 as TSPDiffusionDataset
from utils.two_opt import two_opt_sample



TQDM_KWARGS = {"mininterval": 10.0,  "miniters": 10}

def tqdm(*args, **kwargs):
    return tqdm_orig(*args, **{**TQDM_KWARGS, **kwargs})





class ModelEngine_L1(COMetaModel):
    def __init__(self,
               param_args=None,
               sparse = -1):
        super(ModelEngine_L1, self).__init__(param_args=param_args, node_feature_only=False)
        if sparse > 0:
            param_args = Namespace(**{**vars(self.args), 'sparse_factor': True})
            self.sparse = sparse
        self.args = param_args

    def set_Model(self, ckpt_path, device = 0):
        self.model = TSPModel.load_from_checkpoint(checkpoint_path=ckpt_path, param_args=self.args).to(f'cuda:{device}')

    def set_Data(self, points, batch_size, device = 0):
        self.time_schedule = InferenceSchedule(
            inference_schedule=self.args.inference_schedule,
            T=self.args.diffusion_steps,
            inference_T=self.args.inference_diffusion_steps)
        dataset = TSPDiffusionDataset(points, self.sparse)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.lendataset = len(dataset)

    def test_onelayer(self, xt, sG_count, sG2_times, device = 0, steprange=(0,100), noise_prop = 0.1, total_len = 10000, pbar =None):
        #xt has one more dimension
        edge_index = None

        if self.sparse > 0:
           _, points, edge_index, edge_indicator = next(iter(self.dataloader))
        else:
            _, points = next(iter(self.dataloader))
        B, scale_1extend, _ = points.shape

        scale1_main, scale1_extend = int(total_len//sG_count), scale_1extend
        
        xt_1_list, xt_2_list = [],[]
        batch_idx = 0

        #For less GPU memory with flexible batchsize
        length_xt = xt[0].shape[0] * len(xt)
        if isinstance(xt, list): xt = torch.stack(xt, dim=0)
        xt = xt.view(length_xt// B,  B, scale_1extend, scale_1extend)

        for batch in self.dataloader:
            if self.sparse > 0:
                _, points, edge_index, edge_indicator = batch
            else:
                _, points = batch

            xt_1 = xt[batch_idx]
            assert self.args.parallel_sampling == 1,  'Needs further Coding'
            if self.diffusion_type == 'categorical':   
                xt_1 = (xt_1 > 0).long()

            if self.sparse > 0:
                points = points.reshape(-1, 2)
                edge_index = edge_index.reshape(2, -1)
                src, dst = edge_index
                xt_1 = xt_1[0][src, dst]  

            for i in range(steprange[0], steprange[1]):
                t1, t2 = self.time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                xt_1 = self.model.categorical_denoise_step(points, xt_1, t1, device, edge_index=edge_index, target_t=t2)

            batch_idx += 1
            xt_4view = xt_1[0]

            if noise_prop != 1:
                if B%sG2_times != 0: "the batch size should times of multiscale"
                scale2_main, scale2_extend = scale1_main*sG2_times, scale1_main*sG2_times + 30

                intra_noise_prop = noise_prop
                inter_noise_prop = noise_prop*0.8
                groups     = B // sG2_times

               
                if self.sparse > 0:
                    active_idx = (xt_1 == 1).nonzero(as_tuple=True)[0]
                    src = edge_index[0, active_idx]
                    dst = edge_index[1, active_idx]

                    adj = torch.zeros((scale1_extend, scale1_extend), device=device)
                    adj[src, dst] = 1.0
                    xt_1_list.append(adj)
                else:
                    xt_1 = xt_1[:, :scale1_main, :scale1_main]
                    xt_grouped = xt_1.view(groups, sG2_times, scale1_main, scale1_main)
                    xt_groups = xt2large_noise(groups, xt_grouped, [sG2_times, scale2_main, scale2_extend,],intra_noise_prop, inter_noise_prop) 
                    xt_2_list.append(torch.stack(xt_groups, dim=0).cuda(device=device))
            else:
                xt_2_list.append(xt_1)
        
        if self.sparse > 0:
            groups = 1
            xt_1 = torch.stack(xt_1_list, dim=0)
            xt_1 = xt_1[:, :scale1_main, :scale1_main]
            xt_grouped = xt_1.view(groups, sG2_times, scale1_main, scale1_main)
            xt_groups = xt2large_noise(groups, xt_grouped, [sG2_times, scale2_main, scale2_extend,],intra_noise_prop, inter_noise_prop) 
            xt_2_list.append(torch.stack(xt_groups, dim=0).cuda(device=device))

        
        torch.cuda.empty_cache()
        return xt_2_list
    

    def test_finallayer(self, xt, sG_count, sG2_times, device = 0, steprange=(0,100), total_len = 10000):
        #xt has one more dimension
        edge_index = None

        if self.sparse > 0:
           _, points, edge_index, edge_indicator = next(iter(self.dataloader))
        else:
            _, points = next(iter(self.dataloader))
        B, scale_1extend, _ = points.shape

        scale1_main = int(total_len//len(xt))
        
        xt_2_list = []
        batch_idx = 0

        #For less GPU memory with flexible batchsize
        # xt_stacked = torch.stack(xt, dim=0)
        # ok, oB, oN, _ = xt_stacked.shape
        # xt = xt_stacked.view(ok*oB//B, B , scale_1extend, scale_1extend)
        length_xt = xt[0].shape[0] * len(xt)
        if isinstance(xt, list): 
            xt = [t.to('cuda:0') for t in xt]
            xt_1 = [t[:, :scale1_main, :scale1_main] for t in xt]
            xt_1 = torch.stack(xt_1, dim=0)

        
        xt_grouped = xt_1.view(1, len(xt), scale1_main, scale1_main)

        # 先构造 5 个子图的 block-diagonal (250×250)
        block_diag = torch.block_diag(*[xt_grouped[0, k] for k in range(self.args.GPU_num)])
        xt = torch.stack([block_diag], dim=0).cuda(device=device)
        
        
        xt = xt.view(1,  1, scale_1extend, scale_1extend)

        for batch in self.dataloader:
            if self.sparse > 0:
                _, points, edge_index, edge_indicator = batch
            else:
                _, points = batch

            xt_1 = xt[batch_idx]
            assert self.args.parallel_sampling == 1,  'Needs further Coding'
            if self.diffusion_type == 'categorical':   
                xt_1 = (xt_1 > 0).long()

            if self.sparse > 0:
                points = points.reshape(-1, 2)
                edge_index = edge_index.reshape(2, -1)
                src, dst = edge_index
                xt_1 = xt_1[0][src, dst]  

            for i in range(steprange[0], steprange[1]):
                t1, t2 = self.time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                xt_1 = self.model.categorical_denoise_step(points, xt_1, t1, device, edge_index=edge_index, target_t=t2)

            batch_idx += 1
            xt_4view = xt_1[0]

            xt_2_list.append(xt_1)

        
        torch.cuda.empty_cache()
        return xt_2_list




def test_multilayer_4Sparse(ME3, ME2, total_len, device = 0):
    device = f'cuda:{device}'

    for ss in range(ME3.args.sequential_sampling): 
        _, points = next(iter(ME3.dataloader))
        B, scale1_extend, _ = points.shape

        xt = torch.randn((int(ME3.lendataset//B), B, scale1_extend, scale1_extend), dtype=torch.float32, device = device)
        ms_steps = [int(i* ME3.args.inference_diffusion_steps) for i in ME3.args.multiscale_prop]

        L3_sGcount, L3_sG2_times = ME3.args.L3_clusternum, ME3.args.L32L2_times
        L2_sGcount = ME3.args.L3_clusternum//ME3.args.L32L2_times
        L2_sG2_times = L2_sGcount//ME3.args.GPU_num
        xt = ME3.test_onelayer(xt, sG_count= L3_sGcount, sG2_times= L3_sG2_times, device = device, steprange=(ms_steps[0], ms_steps[1]), noise_prop=0.05, total_len= total_len)
        xt = ME2.test_onelayer(xt, sG_count= L2_sGcount, sG2_times= L2_sG2_times, device = device, steprange=(ms_steps[1], ms_steps[2]), noise_prop=0.01, total_len= total_len)
        
        torch.cuda.empty_cache()
    return xt

def range_preserving_rescale_matrices(matrices, min_val=1e-4, max_val=1.0):
    """
    Keep the original distribution shape, just compress the values to the interval [min_val, max_val].
    linear mapping: v -> scaled_v
    """
    all_vals = np.concatenate([m.flatten() for m in matrices])
    valid_vals = all_vals[all_vals > 0]
    src_min, src_max = valid_vals.min(), valid_vals.max()

    def transform(x):
        x = np.clip(x, src_min, src_max)
        return (x - src_min) / (src_max - src_min) * (max_val - min_val) + min_val

    return [transform(m) for m in matrices]

def process_test_end_heatmap_4Sparse(xt, edge_index_list, sequential_sampling, cluster_global_indices, main_points, merge_points, 
                             merge_thr = 1000, parallel_sampling = 1, test_2opt_iterations =1000, sparser = False, excute_num=8):
    small_adjs, all_points, offset, cluster_map = [], [], 0, []
    base_scale = len(main_points[0])
    subgraph_num = len(cluster_global_indices) 

    assert sequential_sampling == 1, "SS > 1 not yet supported"
    all_src, all_dst, all_val =[],[],[]

    for ss in range(sequential_sampling):
        for idx in range(subgraph_num):
            xt_sub = xt[idx].detach().cpu()           # shape [E]
            edge_index = edge_index_list[idx][0].detach().cpu()  # shape [2, E]

            # cutting
            src_all, dst_all = edge_index
            mask = (src_all < base_scale) & (dst_all < base_scale)  # 使用 Tensor 操作，更高效
            edge_index_sub = edge_index[:, mask]  # 筛选后的边，shape [2, E_sub]
            xt_sub_selected = xt_sub[mask] 

            small_adj = torch.sparse_coo_tensor(
                indices=edge_index_sub,
                values=xt_sub_selected,
                size=(base_scale, base_scale)
            ).to_dense().cpu().numpy()

            small_adjs.append(small_adj)
            pts = merge_points[cluster_global_indices[idx]]   
            all_points.append(pts)

            cluster_map.append((offset, offset + base_scale))
            offset += base_scale

        small_adjs = range_preserving_rescale_matrices(small_adjs, min_val=1e-4, max_val=1.0)

        big_size = subgraph_num * base_scale
        big_adj  = np.zeros((big_size, big_size), dtype=small_adjs[0].dtype)  + 1e-5  #So far with original adding and 1500 mergethreshold may have a good performance
        for g, mat in enumerate(small_adjs):
            s = g * base_scale
            big_adj[s:s+base_scale, s:s+base_scale] = mat
        big_adj = big_adj  
        big_adj = big_adj*10000


        big_points = np.vstack(all_points)
        mapping = np.concatenate(cluster_global_indices, axis=0)

        # Note that merge_tours needs adj_mat with (Sequential, big_size, big_size)
        tours, merge_iterations = merge_tours_v4(
            adj_mat       = big_adj[None, :, :],
            np_points = big_points,
            merge_thr = merge_thr,
            parallel_sampling = parallel_sampling,
        )

        solved_tour, ns = two_opt_sample(
            big_points.astype("float64"),
            np.array(tours).astype("int64"),
            test_2opt_iterations,
            excute_num
        )

        final_tour = solved_tour[0].tolist()
        final_tour1 = mapping[final_tour]
        
    return final_tour1

def process_dividecluster(global_coord, L3sG_count, L3sG_label, scale2_times = 5):
    L2sG_label = np.full(len(global_coord), -1, dtype=int)
    labellist = list(set(L3sG_label))
    for i in range(L3sG_count):
        L3sG_idx = np.where(L3sG_label == labellist[i])[0]
        L3sG_coord = global_coord[L3sG_idx]  
        L2sG_label_sG, L2sG_center = balanced_kmeans_mcmf_fast_v3(L3sG_coord, scale2_times, m=3)
        for j in range(scale2_times):
            L2sG_label_glob = labellist[i] * scale2_times + j
            L2sG_label[L3sG_idx[L2sG_label_sG == j]] = L2sG_label_glob

    return L2sG_label


def Threadworker(rank, ME3, ME2, data_chunk3, data_chunk2, total_len, return_dict):
    set_global_seed(ME3.args.random_seed)
    ME3.set_Data(data_chunk3, batch_size=ME3.args.L3_batchsize, device=rank)
    ME2.set_Data(data_chunk2, batch_size=ME3.args.L2_batchsize, device=rank)
    # ME3.set_Data(data_chunk3, batch_size=1 , device=rank)

    with torch.no_grad():
        result = test_multilayer_4Sparse(ME3, ME2, total_len, device=rank)
        return_dict[rank] = result


def parallel_Threadworker(ME3_list, ME2_list, ME1_list,  L3_enhanced_data, L2_enhanced_data, L1_enhanced_data, total_len):
    GPU_num =  ME3_list[0].args.GPU_num
    chunk_size = len(L3_enhanced_data) // GPU_num
    datachunks_3 = [L3_enhanced_data[i * chunk_size : (i + 1) * chunk_size] for i in range(GPU_num)]
    chunk_size = len(L2_enhanced_data) // GPU_num
    datachunks_2 = [L2_enhanced_data[i * chunk_size : (i + 1) * chunk_size] for i in range(GPU_num)]

    threads = []
    return_list = [None] * GPU_num
    for i in range(GPU_num):
        t = threading.Thread(target=Threadworker, args=(i, ME3_list[i], ME2_list[i], datachunks_3[i], datachunks_2[i], 
                                                        total_len, return_list))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    L1_xt = []

    for i in range(GPU_num):
        adj_list = return_list[i]  #list of tensors
        L1_xt.extend(adj_list)

    ME1 = ME1_list[0]
    ME1.set_Data(L1_enhanced_data, batch_size=1 , device=0)
    ms_steps = [int(i* ME1.args.inference_diffusion_steps) for i in ME1.args.multiscale_prop]
    with torch.no_grad():
        xt = ME1.test_finallayer(L1_xt, sG_count=1, sG2_times=1, device = 0, steprange=(ms_steps[2], ms_steps[3]), total_len= total_len)
    
    for batch in ME1.dataloader:
        _, points, edge_index, edge_indicator = batch
    return xt, [edge_index]


def load_and_preprocess_batch(batch_idx, dataset):
    starttime = time.time()

    _, nodes, _, _ = dataset
    global_coord = nodes.squeeze(0).numpy() * 10000

    #Balanced_Kmeans By layers:
    L1sG_count = int(args.L3_clusternum/args.L32L2_times/args.L22L1_times)
    L1sG_label, L1sG_center = balanced_kmeans_mcmf_fast_v3(global_coord, L1sG_count)
    if args.if_plot_clusters:   plot_clusters(global_coord, L1sG_label, L1sG_center)

    L2sG_label = process_dividecluster(global_coord, L1sG_count, L1sG_label, args.L22L1_times)
    L2sG_count = int(args.L3_clusternum/args.L32L2_times)
    L3sG_label = process_dividecluster(global_coord, L2sG_count, L2sG_label, args.L32L2_times)


    #LKH solution For Layer1, but efficient for large subgraphs.
    L1_tour = [0]

    L2_super_coords = build_coord_dict(
        [global_coord[L2sG_label == i].mean(axis=0) for i in range(L2sG_count)], prefix="super" )
    solver = elkai.Coordinates2D(L2_super_coords)
    L2_tour = [int(node.split('_')[-1]) for node in solver.solve_tsp()[:-1]]

    # L2_tour = []
    # for node_i in L3_tour:
    #     for node_j in range(args.scale2_times):
    #         L2_tour.append(node_i * args.scale2_times + node_j)

    L3_tour = []
    for node_i in L2_tour:
        for node_j in range(args.L32L2_times):
            L3_tour.append(node_i * args.L32L2_times + node_j)
    # print("LKH time", time.time() - starttime)

    L1_enhanced_data, L1_cluster_global_indices, L1_main_points =  process_clusters_wotour(L1_tour, L1sG_label, global_coord)
    L2_enhanced_data, L2_cluster_global_indices, L2_main_points =  process_clusters_wtour(L2_tour, L2sG_label, global_coord, redundancy_length = 8, bridge_length=14)
    L3_enhanced_data, L3_cluster_global_indices, L3_main_points =  process_clusters_wtour(L3_tour, L3sG_label, global_coord, redundancy_length = 3, bridge_length= 4)
    
    print(batch_idx, "-th Dividing Cost time", time.time() - starttime)

    return [L3_enhanced_data, L2_enhanced_data, L1_enhanced_data, global_coord, L1sG_label, L1sG_center, L1_tour, L1_cluster_global_indices, L1_main_points]



def run_diffusion_on_gpu(batch_idx, data, ME3_list, ME2_list, ME1_list):
    starttime = time.time()
    [L3_enhanced_data, L2_enhanced_data, L1_enhanced_data] = data[:3]
    len_tour = len(data[3])
    solution_adjmatrix, solution_points = parallel_Threadworker(ME3_list, ME2_list, ME1_list, L3_enhanced_data, L2_enhanced_data, L1_enhanced_data, len_tour)
    print(batch_idx,"-th Diffusing cost time", time.time() - starttime)
    return (solution_adjmatrix, solution_points)



def append_array_to_list_dir(list_dir, array):
    os.makedirs(list_dir, exist_ok=True)
    idx = len(os.listdir(list_dir))
    filename = os.path.join(list_dir, f"{idx:05d}.npy")
    np.save(filename, array)

def decode_and_store_result(batch_idx, args, data, result):
    solution_adjmatrix, solution_points  = result
    global_coord, L3sG_label, L3sG_center, L3_tour, cluster_global_indices3, main_points3 = data[3:]
    starttime = time.time()
    # global_tour_indices = process_test_end(solution_adjmatrix, solution_points, args.sequential_sampling, cluster_global_indices2, main_points2)
    global_tour_indices = process_test_end_heatmap_4Sparse(solution_adjmatrix, solution_points, args.sequential_sampling, cluster_global_indices3, main_points3, global_coord, 
                                                   merge_thr=args.merge_threshold, test_2opt_iterations= args.opt_iter, excute_num = args.decoexcute_num)
    print(batch_idx, '-th Decoding cost time:', time.time() - starttime)


    #Temporary Storage
    append_array_to_list_dir('storage/temp_results/tour_solutions', global_tour_indices)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

async def async_main(args, whole_dataset, ME3_list, ME2_list, ME1_list):
    set_global_seed(args.random_seed)
    total = len(whole_dataset)

    # Two queues：loader->gpu, gpu->decoder
    q1 = asyncio.Queue(maxsize=2)
    q2 = asyncio.Queue(maxsize=2) 

    loop = asyncio.get_event_loop()
    loader_executor = ThreadPoolExecutor(max_workers=args.loadexcute_num)
    gpu_lock = asyncio.Lock()
    decode_executor = ThreadPoolExecutor(max_workers=args.decoexcute_num)

    # Stage 1: Loader (Divide)
    async def loader():
        for b in range(total):
            data = await loop.run_in_executor(loader_executor, load_and_preprocess_batch, b, whole_dataset[b])
            await q1.put((b, data))
            # q1.maxsize=2 会在 “窗口” 超过2时自动阻塞，保证后续只跑到 N+1
        await q1.put(None)  # 结束标记

    # Stage 2: GPU (Diffuse)
    async def gpu_worker():
        while True:
            item = await q1.get()
            if item is None:
                await q2.put(None)
                break
            b, data = item
            async with gpu_lock:
                result = run_diffusion_on_gpu(b, data, ME3_list, ME2_list, ME1_list)
            await q2.put((b, data, result))

    # Stage 3: Decoder
    async def decoder():
        while True:
            item = await q2.get()
            if item is None:
                break
            b, data, result = item
            await loop.run_in_executor(decode_executor, decode_and_store_result, b, args, data, result)

    await asyncio.gather(loader(), gpu_worker(), decoder())




def main(args):
    #Initialization
    args.multiscale_prop = ast.literal_eval(args.multiscale_prop)
    setup_logging(args, args.logfile_path ,logging.INFO if not args.debug else logging.DEBUG)

    for filename in os.listdir('storage/temp_results/tour_solutions'):
        file_path = os.path.join('storage/temp_results/tour_solutions', filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path) 

    starttime = time.time()
    L3ME_base = ModelEngine_L1(args)
    L3ME_base.set_Model(args.L3_model_ckptpath)
    L2ME_base = ModelEngine_L1(args, sparse=args.L2_sparse)
    L2ME_base.set_Model(args.L2_model_ckptpath)
    L1ME_base = ModelEngine_L1(args, sparse=args.L1_sparse)
    L1ME_base.set_Model(args.L1_model_ckptpath)

    # 为每张卡 clone 一份模型
    L3ME_list = [copy.deepcopy(L3ME_base).to(f'cuda:{i}') for i in range(args.GPU_num)]
    L2ME_list = [copy.deepcopy(L2ME_base).to(f'cuda:{i}') for i in range(args.GPU_num)]
    L1ME_list = [L1ME_base]
    print("Set Model time", time.time() - starttime)

    
    whole_dataset = TSPGraphDataset(args.data_path, sparse_factor=-1)
    
    
    asyncio.run(async_main(args, whole_dataset, L3ME_list, L2ME_list, L1ME_list))
    logging.info(f"Average time for {len(whole_dataset)} using time: {((time.time() - starttime) / len(whole_dataset))}")


    list_dir = os.path.join('storage/temp_results', "tour_solutions")
    idx = 0
    avg_gap = 0
    success_time = len(whole_dataset)
    for fname in sorted(os.listdir(list_dir)):
        if fname.endswith(".npy"):
            _, nodes, _, GT_tour = whole_dataset[idx] 
            global_coord = nodes.squeeze(0).numpy() * 10000

            solved_tour_indices = np.load(os.path.join(list_dir, fname))
            solved_cost = compute_tour_cost(global_coord, solved_tour_indices)
            
            if args.GT_had: 
                GT_cost = compute_tour_cost(global_coord, GT_tour)
                gap = (solved_cost/GT_cost - 1)*100
                if gap > 0:   avg_gap += gap
                else:         success_time -= 1
                logging.info(f"{idx}-th Graph with Diffusion, tour cost: {solved_cost:.2f}, GT cost: {GT_cost:.2f}, with GAP = {gap:.3f}%")
            else: 
                logging.info(f"{idx}-th Graph with Diffusion, tour cost: {solved_cost:.2f}")
            idx += 1
            
            # L3sG_label, L3sG_center = balanced_kmeans_mcmf_fast_v3(global_coord, 1)
            # plot_global(global_coord, L3sG_label, np.array(solved_tour_indices),  L3sG_center, save_path='./figures/global_tsp_result_diff_3.png')
            # time.sleep(5)

    if args.GT_had: logging.info(f"Average gap for {success_time} is {(avg_gap/success_time):.3f}%")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Large Scale TSP with LKH")

    # Scale Parameters
    parser.add_argument("--redundancy_length", type=int, default=3, help="Number of nodes to be added in one side of target subgraph")
    parser.add_argument("--bridge_length", type=int, default=4, help="Number of nodes to be added for cycle")
    parser.add_argument("--L3_clusternum", type=int, default=20, help="Number of clustered subgraphs") #200/40
    parser.add_argument('--L3_batchsize', type=int, default=20)
    parser.add_argument('--L2_batchsize', type=int, default=2)
    parser.add_argument("--L32L2_times", type=int, default=5)
    parser.add_argument("--L22L1_times", type=int, default=4)
    parser.add_argument("--pre_prop", type=int, default=0.8)
    parser.add_argument("--noise_prop", type=int, default=0.03)
    parser.add_argument('--multiscale', default=True)
    parser.add_argument('--random_seed', type=int, default=42)
    

    # Diffusion Parameters
    parser.add_argument("--L3_model_ckptpath", type=str, default="storage/ckpt/TSP60_EP50.ckpt", help="Path to the checkpoint")
    parser.add_argument("--L2_model_ckptpath", type=str, default="storage/ckpt/TSP280_EP50.ckpt", help="Path to the checkpoint")
    parser.add_argument("--L2_sparse", type=int, default=-1)
    parser.add_argument("--L1_model_ckptpath", type=str, default="storage/ckpt/epoch=32.ckpt", help="Path to the checkpoint")
    parser.add_argument("--L1_sparse", type=int, default=100)
    parser.add_argument("--multiscale_prop", type=str, default='[0,0.8,0.9,1]', help="Path to the checkpoint")
    
    parser.add_argument('--diffusion_type', type=str, default='categorical')
    parser.add_argument('--diffusion_schedule', type=str, default='cosine')
    parser.add_argument('--diffusion_steps', type=int, default=200)
    parser.add_argument('--inference_diffusion_steps', type=int, default=40)
    parser.add_argument('--inference_schedule', type=str, default='cosine')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)
   
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)
    parser.add_argument('--save_numpy_heatmap', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true', default=False)

    # Plotting Parameters
    parser.add_argument("--if_plot_clusters", default=False, action="store_true", help="Plot the clustered points")
    parser.add_argument("--if_plot_subgraph", default=True, action="store_true", help="Plot the cluster subgraph tour")
    parser.add_argument("--if_plot_global", default=True, action="store_true", help="Plot the global TSP tour")
    #Paths regarding data and figure storage
    parser.add_argument("--logfile_path", type=str, default="./storage/logs/tsp_large_LKH_v87.log", help="Path to the log file")
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    #figure storage is not added because it is not used in the main function
    parser.add_argument("--data_path", type=str, default="./storage/data/tsp/tsp1000_test_concorde_16item.txt", help="Path to the TSP dataset")

    #Parallel
    # parser.add_argument('--Asynch_time', type=int, default=4)
    parser.add_argument('--GPU_num', type=int, default=2)
    parser.add_argument('--GT_had', action='store_true')
    parser.add_argument('--loadexcute_num', type=int, default=8)
    parser.add_argument('--decoexcute_num', type=int, default=24)
    parser.add_argument('--merge_threshold', type=int, default=1500)
    parser.add_argument('--opt_iter', type=int, default=1000)

    # 解析参数
    args = parser.parse_args()

    # 调用主函数
    main(args)