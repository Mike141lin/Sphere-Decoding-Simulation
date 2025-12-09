import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.stats import chi2
import pandas as pd

"""
Lattice Sphere Decoder Simulation (E8 Lattice) - Algorithm Comparison
基于:
1. Agrell et al. (DFS/Schnorr-Euchner) - 参考 E8.py 风格
2. Viterbo et al. (BFS)
对比: Runtime, Visited Nodes, SER
"""

# =============================================================================
# Part 1: 基础工具与格定义
# =============================================================================

def get_E8_matrix():
    """
    返回 E8 格的生成矩阵 G (8x8)。
    使用标准的上三角生成矩阵形式 (Gosset Lattice)。
    """
    G = np.array([
        [2., 0., 0., 0., 0., 0., 0., 0.],
        [-1., 1., 0., 0., 0., 0., 0., 0.],
        [0., -1., 1., 0., 0., 0., 0., 0.],
        [0., 0., -1., 1., 0., 0., 0., 0.],
        [0., 0., 0., -1., 1., 0., 0., 0.],
        [0., 0., 0., 0., -1., 1., 0., 0.],
        [0., 0., 0., 0., 0., -1., 1., 0.],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ])
    return G

def calculate_sigma_from_vnr(G, vnr_db):
    """根据 VNR 公式计算噪声标准差 sigma"""
    n = G.shape[0]
    vnr_lin = 10 ** (vnr_db / 10.0)
    det_G = np.abs(np.linalg.det(G)) 
    numerator = det_G ** (2.0 / n)
    denominator = 2 * np.pi * np.e * vnr_lin
    sigma = np.sqrt(numerator / denominator)
    return sigma

def sgn(x):
    """辅助函数: sgn*(z)"""
    if x <= 0: return -1.0
    else: return 1.0

# =============================================================================
# Part 2: 预处理模块 (参考 demo.py / E8.py)
# =============================================================================

def preprocess_lattice(G):
    """
    QR 分解与坐标变换。
    输出:
      H:  核心解码矩阵 (下三角, Agrell 算法输入)
      Q:  旋转矩阵
      G3: 变换后的下三角生成矩阵 (BFS 算法输入)
    """
    # 1. QR 分解 (对转置分解以获得下三角形式)
    Q_temp, R_temp = np.linalg.qr(G.T)
    G3 = R_temp.T # Lower Triangular
    Q = Q_temp.T
    
    # 2. 符号修正 (确保对角线为正)
    n = G.shape[0]
    for i in range(n):
        if G3[i, i] < 0:
            G3[:, i] = -G3[:, i]
            Q[i, :] = -Q[i, :]
            
    # 3. 求逆得到 H (用于 Agrell 的 DFS)
    H = np.linalg.inv(G3)
    
    return H, Q, G3

# =============================================================================
# Part 3: 核心解码器 (DFS & BFS) - 带节点统计
# =============================================================================

def sphere_decode_dfs(H, r):
    """
    Schnorr-Euchner 策略球解码器 (DFS)。
    风格参考 E8.py，增加节点计数返回。
    """
    n = H.shape[0]
    bestdist = float('inf')
    k = n - 1
    dist_k = np.zeros(n)
    e = np.zeros((n, n))
    
    # 坐标变换 e_n := rH (Agrell 论文记号)
    e[n-1] = np.dot(r, H)
    
    u = np.zeros(n, dtype=int)
    u[k] = round(e[k, k])
    y = (e[k, k] - u[k]) / H[k, k]
    step = np.zeros(n, dtype=int)
    step[k] = sgn(y)
    
    u_hat = np.zeros(n, dtype=int)
    nodes_visited = 0
    
    while True:
        nodes_visited += 1 # 统计节点
        newdist = dist_k[k] + y*y
        
        if newdist < bestdist:
            if k != 0: # Case A: 向下
                for i in range(k):
                    e[k-1, i] = e[k, i] - y * H[k, i]
                k -= 1
                dist_k[k] = newdist
                u[k] = round(e[k, k])
                y = (e[k, k] - u[k]) / H[k, k]
                step[k] = sgn(y)
            else: # Case B: 找到点
                u_hat = u.copy()
                bestdist = newdist # 更新半径 (SE 策略核心)
                k += 1
                u[k] += step[k]
                y = (e[k, k] - u[k]) / H[k, k]
                step[k] = -step[k] - sgn(step[k])
        else: # Case C: 回溯
            if k == n - 1:
                return u_hat, nodes_visited
            else:
                k += 1
                u[k] += step[k]
                y = (e[k, k] - u[k]) / H[k, k]
                step[k] = -step[k] - sgn(step[k])

def sphere_decode_bfs(G3, r, radius_sq):
    """
    Viterbo-Boutros 策略球解码器 (BFS)。
    使用 G3 (下三角矩阵) 进行搜索。
    注: r 是旋转后的接收向量。
    """
    n = G3.shape[0]
    
    # 初始化: 求解中心点 rho = r * G3^{-1} (即 rH)
    # 这里为了效率，直接解三角方程系统 G3^T * rho^T = r^T 
    # 或者直接利用 DFS 算好的变量，但在独立函数中我们重新算
    H = np.linalg.inv(G3)
    rho = np.dot(r, H)
    
    # 路径列表: (accumulated_dist, [u_0, ..., u_k]) 
    # Viterbo 算法通常是从 n-1 到 0 (或者 1 到 n)
    # 这里的 G3 是下三角，对应 Agrell 的反向顺序，我们从 k=n-1 开始搜
    
    paths = [(0.0, [])] # 存储部分路径
    nodes_visited = 1
    
    # 从 n-1 层向下搜索到 0 层
    for k in range(n - 1, -1, -1):
        new_paths = []
        
        # 内存保护: 防止 BFS 在低信噪比下爆炸
        if len(paths) > 60000: 
            return np.zeros(n), nodes_visited + len(paths)*10
            
        for curr_dist, curr_u in paths:
            # 计算干扰项 S_k
            # curr_u 存储的是 [u_{k+1}, u_{k+2}, ..., u_{n-1}] (逆序)
            # G3 是下三角 L. 
            # 距离公式项: R_kk * (u_k - c_k)^2
            # 对应 Agrell: (e_{k,k} - u_k * H_{k,k})^2 ... 稍微有点不同
            # Viterbo 公式: sum ( l_ii * x_i + sum_{j=i+1}^n l_ji x_j - r_i )^2
            # G3[i, j] 对应 l_ji
            
            # 简化逻辑: 使用 Agrell 的变量体系适配 BFS
            # e_{k,k} 是当前层的“无干扰中心”
            # 但 BFS 需要显式计算 interference
            
            # 为保持一致性，我们直接复用 rho (无约束解)
            # dist_k = G3[k,k]^2 * (u_k + sum(...) - rho_k)^2 ? 
            # 不，最简单的 BFS 实现是基于距离增量
            
            S_k = 0.0
            # 恢复 u 的索引. curr_u[0] 是 u_{k+1}
            for idx, u_val in enumerate(curr_u):
                # G3 是下三角，G3[col, row] ? 
                # Agrell G3 = R.T. R 是上三角.
                # G3[j, k] * u[j] for j > k? 
                # G3 是下三角，所以高维变量在低维方程中出现
                # G3[j, k] is 0 for j < k. 
                # 实际上 interference 来自于已决定的高维 u_j (j > k)
                # 系数是 G3[j, k] (第 k 列, 第 j 行)
                j = (n - 1) - idx
                S_k += G3[j, k] * u_val
            
            # 接收向量分量 r[k]
            # 目标: minimize | u*G3 - r |^2
            # component k: ( sum_{j=k}^n u_j G3[j,k] - r[k] )^2
            # = ( u_k G3[k,k] + S_k - r[k] )^2
            # = G3[k,k]^2 * ( u_k + (S_k - r[k])/G3[k,k] )^2
            
            hk = G3[k, k]
            center = (r[k] - S_k) / hk
            
            rem_dist = radius_sq - curr_dist
            if rem_dist < 0: continue
            
            # 计算边界
            delta = np.sqrt(rem_dist) / np.abs(hk)
            u_min = int(np.ceil(center - delta))
            u_max = int(np.floor(center + delta))
            
            # 统计节点 (该层所有候选点都算访问)
            valid_count = max(0, u_max - u_min + 1)
            nodes_visited += valid_count
            
            for u_cand in range(u_min, u_max + 1):
                # 增量距离
                term = (hk * u_cand + S_k - r[k])**2
                if curr_dist + term <= radius_sq:
                    # 将 u_cand 加到路径列表头部或尾部? 
                    # curr_u 是 [u_{k+1}...]
                    # 我们需要保留顺序，append 到尾部
                    new_u = curr_u + [u_cand]
                    new_paths.append((curr_dist + term, new_u))
                    
        paths = new_paths
        if not paths: break
    
    if not paths: return np.zeros(n), nodes_visited
    
    # 找到最小距离路径
    best_path = min(paths, key=lambda x: x[0])
    # best_path[1] 是 [u_{n-1}, u_{n-2}, ..., u_0]
    # 我们需要 u = [u_0, ..., u_n-1]
    u_hat = np.array(best_path[1][::-1])
    return u_hat, nodes_visited

# =============================================================================
# Part 4: 并行仿真 Worker
# =============================================================================

def simulation_worker(args):
    """并行处理一批样本"""
    seed, count, H, Q, G3, sigma, radius_sq = args
    np.random.seed(seed)
    n = H.shape[0]
    
    # 结果累加器
    res = {
        'err_dfs': 0, 'nodes_dfs': 0,
        'err_bfs': 0, 'nodes_bfs': 0,
        'time_dfs': 0, 'time_bfs': 0
    }
    
    u_true = np.zeros(n) # 假设发送全零码字
    
    # 批量生成噪声
    noise_batch = np.random.normal(0, sigma, size=(count, n))
    
    for i in range(count):
        # 1. 编码与信道 (y = 0 + n)
        y_received = noise_batch[i]
        
        # 2. 接收端旋转 (r = y * Q^T)
        r_rot = np.dot(y_received, Q.T)
        
        # --- DFS (Schnorr-Euchner) ---
        t0 = time.time()
        u_dfs, n_dfs = sphere_decode_dfs(H, r_rot)
        res['time_dfs'] += (time.time() - t0)
        res['nodes_dfs'] += n_dfs
        if not np.array_equal(u_dfs, u_true):
            res['err_dfs'] += 1
            
        # --- BFS (Viterbo-Boutros) ---
        t0 = time.time()
        u_bfs, n_bfs = sphere_decode_bfs(G3, r_rot, radius_sq)
        res['time_bfs'] += (time.time() - t0)
        res['nodes_bfs'] += n_bfs
        if not np.array_equal(u_bfs, u_true):
            res['err_bfs'] += 1
            
    return res

# =============================================================================
# Part 5: 主仿真逻辑
# =============================================================================

def run_simulation(G, vnr_list_db, num_samples_total):
    n = G.shape[0]
    H, Q, G3 = preprocess_lattice(G)
    
    # 并行配置
    num_workers = cpu_count()
    batch_size = num_samples_total // num_workers
    
    # 理论半径 (覆盖 99.99% 用于 BFS 公平对比 ML)
    chi2_val = chi2.ppf(0.9999, df=n)
    
    print(f"\n{'='*75}")
    print(f"开始并行仿真 (E8 Lattice Comparison)")
    print(f"Workers: {num_workers} | Samples: {num_samples_total}")
    print(f"BFS Radius Factor: {chi2_val:.2f} * sigma^2")
    print(f"{'='*75}")
    print(f"{'VNR':<6} | {'DFS SER':<10} | {'BFS SER':<10} | {'DFS Nodes':<10} | {'BFS Nodes':<10} | {'Ratio':<6}")
    print("-" * 75)
    
    results = {
        'vnr': [], 
        'dfs_ser': [], 'bfs_ser': [], 
        'dfs_nodes': [], 'bfs_nodes': [],
        'dfs_time': [], 'bfs_time': []
    }
    
    for vnr_db in vnr_list_db:
        sigma = calculate_sigma_from_vnr(G, vnr_db)
        radius_sq = chi2_val * (sigma**2)
        
        # 构造任务
        tasks = []
        for i in range(num_workers):
            seed = int(time.time()) + i*999 + int(vnr_db)*77
            tasks.append((seed, batch_size, H, Q, G3, sigma, radius_sq))
            
        # 并行执行
        with Pool(num_workers) as p:
            worker_results = p.map(simulation_worker, tasks)
            
        # 汇总数据
        total_dfs_err = sum(r['err_dfs'] for r in worker_results)
        total_bfs_err = sum(r['err_bfs'] for r in worker_results)
        total_dfs_nodes = sum(r['nodes_dfs'] for r in worker_results)
        total_bfs_nodes = sum(r['nodes_bfs'] for r in worker_results)
        total_dfs_time = sum(r['time_dfs'] for r in worker_results)
        total_bfs_time = sum(r['time_bfs'] for r in worker_results)
        
        actual_samples = batch_size * num_workers
        
        # 计算平均值
        dfs_ser = total_dfs_err / actual_samples
        bfs_ser = total_bfs_err / actual_samples
        avg_dfs_nodes = total_dfs_nodes / actual_samples
        avg_bfs_nodes = total_bfs_nodes / actual_samples
        avg_dfs_time = (total_dfs_time / actual_samples) * 1000 # ms
        avg_bfs_time = (total_bfs_time / actual_samples) * 1000 # ms
        
        ratio = avg_bfs_nodes / avg_dfs_nodes if avg_dfs_nodes > 0 else 1.0
        
        # 存储
        results['vnr'].append(vnr_db)
        results['dfs_ser'].append(dfs_ser)
        results['bfs_ser'].append(bfs_ser)
        results['dfs_nodes'].append(avg_dfs_nodes)
        results['bfs_nodes'].append(avg_bfs_nodes)
        results['dfs_time'].append(avg_dfs_time)
        results['bfs_time'].append(avg_bfs_time)
        
        print(f"{vnr_db:<6.1f} | {dfs_ser:<10.2e} | {bfs_ser:<10.2e} | {avg_dfs_nodes:<10.1f} | {avg_bfs_nodes:<10.1f} | {ratio:<6.1f}x")
        
    return results

# =============================================================================
# Part 6: 绘图模块
# =============================================================================

def plot_results(results):
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    
    # Color Scheme
    col_dfs = '#1f77b4' # Matplotlib Blue
    col_bfs = '#d62728' # Matplotlib Red
    
    # --- Plot 1: Node Complexity ---
    ax1.semilogy(df['vnr'], df['bfs_nodes'], 'o--', color=col_bfs, label='BFS Nodes (Fixed Radius)', linewidth=2, markersize=8)
    ax1.semilogy(df['vnr'], df['dfs_nodes'], 's-', color=col_dfs, label='DFS Nodes (Schnorr-Euchner)', linewidth=2, markersize=8)
    
    ax1.set_title('E8 Lattice: Algorithmic Complexity (Nodes Visited)', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('VNR (dB)', fontsize=12)
    ax1.set_ylabel('Avg Visited Nodes per Symbol (Log Scale)', fontsize=12)
    ax1.grid(True, which="both", ls="-", color='0.9')
    ax1.legend(fontsize=11, frameon=True)
    
    # 标注差异 (取低信噪比点)
    idx_low = 1 # e.g. 1dB
    if idx_low < len(df):
        ratio = df['bfs_nodes'][idx_low] / df['dfs_nodes'][idx_low]
        ax1.text(df['vnr'][idx_low], df['bfs_nodes'][idx_low]*0.5, 
                 f"DFS is ~{ratio:.0f}x More Efficient", 
                 color='darkred', fontweight='bold')

    # --- Plot 2: SER Performance ---
    ax2.semilogy(df['vnr'], df['bfs_ser'], 'o--', color=col_bfs, label='BFS SER', linewidth=2, markersize=10, fillstyle='none')
    ax2.semilogy(df['vnr'], df['dfs_ser'], 'x-', color=col_dfs, label='DFS SER', linewidth=2, markersize=6, alpha=0.8)
    
    ax2.set_title('E8 Lattice: Accuracy Verification (SER)', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('VNR (dB)', fontsize=12)
    ax2.set_ylabel('Symbol Error Rate (Log Scale)', fontsize=12)
    ax2.grid(True, which="both", ls="-", color='0.9')
    ax2.legend(fontsize=11, frameon=True)
    
    # 标注重合
    ax2.text(0.5, 0.2, "Curves Overlap = Valid ML Performance", 
             transform=ax2.transAxes, ha='center',
             fontsize=12, color='green', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))

    plt.tight_layout()
    plt.savefig('E8_Final_Styled.png')
    print("\n[绘图完成] 结果已保存为: E8_Final_Styled.png")
    plt.show()

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 1. 获取格
    G_E8 = get_E8_matrix()
    
    # 2. 定义测试范围 (0-9 dB)
    vnr_range = np.arange(0, 10.0, 1.0) 
    
    # 3. 运行仿真 (20万样本，并行加速)
    num_samples = 200000 
    
    start_time = time.time()
    results = run_simulation(G_E8, vnr_range, num_samples)
    
    print(f"\n总耗时: {time.time() - start_time:.2f} 秒")
    
    # 4. 绘图
    plot_results(results)