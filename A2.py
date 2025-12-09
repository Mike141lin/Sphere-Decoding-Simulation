import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# 1. 基础工具与 A2 定义
# =============================================================================

def get_A2_generator():
    """返回 A2 格的生成矩阵"""
    # A2 格基向量: [1, 0] 和 [0.5, sqrt(3)/2]
    G = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3.0)/2.0]
    ])
    return G

def calculate_sigma(G, vnr_db):
    """根据 VNR (dB) 计算噪声标准差 sigma"""
    n = G.shape[0]
    vnr_lin = 10 ** (vnr_db / 10.0)
    # 计算行列式 (体积)
    det_G = np.abs(np.linalg.det(G))
    # VNR 公式: VNR = (det_G)^(2/n) / (2 * pi * e * sigma^2)
    numerator = det_G ** (2.0 / n)
    denominator = 2 * np.pi * np.e * vnr_lin
    sigma = np.sqrt(numerator / denominator)
    return sigma

# =============================================================================
# 2. Optimal Coset Decoder (A2 专用) - Ground Truth
# =============================================================================

def decode_optimal_a2(G_inv, y_received):
    """
    A2 格的最优陪集解码器 (O(1) 复杂度)。
    原理：直接在斜坐标系中寻找最近的整数点。
    """
    # 1. 转换到晶格坐标系: u_continuous = y * G^-1
    u_cont = y_received @ G_inv
    
    # 2. 找到参考整数点 (向下取整)
    u_floor = np.floor(u_cont).astype(int)
    
    # 3. 候选点：(0,0), (0,1), (1,0), (1,1) 的偏移
    candidates_offset = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    
    best_dist = float('inf')
    best_u = None
    
    # 4. 暴力检查这 4 个邻居 (转换回欧氏空间测距)
    # G_inv 的逆是 G
    G = np.linalg.inv(G_inv)
    
    for offset in candidates_offset:
        u_test = u_floor + offset
        # 映射回欧氏空间: x_test = u_test * G
        x_test = u_test @ G
        
        # 计算距离平方
        d_sq = np.sum((x_test - y_received)**2)
        if d_sq < best_dist:
            best_dist = d_sq
            best_u = u_test
            
    return best_u

# =============================================================================
# 3. 预处理 (QR 分解) - 用于 DFS 和 BFS
# =============================================================================

def preprocess_lattice(G):
    """
    QR 分解: G = H * Q (这里 H 为下三角)
    """
    Q_temp, R_temp = np.linalg.qr(G.T)
    H = R_temp.T # 下三角矩阵
    Q_rot = Q_temp.T # 旋转矩阵
    
    # 确保对角线为正
    n = G.shape[0]
    for i in range(n):
        if H[i, i] < 0:
            H[:, i] = -H[:, i]
            Q_rot[i, :] = -Q_rot[i, :]
            
    return H, Q_rot

# =============================================================================
# 4. DFS (Schnorr-Euchner) - 深度优先
# =============================================================================

def sgn(x):
    return 1 if x >= 0 else -1

def decode_dfs_se(H, r_rot):
    """
    Schnorr-Euchner 策略: 深度优先，自适应半径
    """
    n = H.shape[0]
    bestdist = float('inf')
    k = n - 1
    dist = np.zeros(n)
    E = np.zeros((n, n))
    E[n-1] = r_rot
    u = np.zeros(n, dtype=int)
    step = np.zeros(n, dtype=int)
    
    # 初始 Babai 点
    u[k] = int(round(E[k][k] / H[k][k]))
    y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
    step[k] = sgn(y)
    
    u_hat = np.zeros(n, dtype=int)

    while True:
        d_k = (E[k][k] - u[k] * H[k][k])**2
        newdist = dist[k] + d_k
        
        if newdist < bestdist:
            if k != 0: # 向下
                E[k-1, :k] = E[k, :k] - u[k] * H[k, :k]
                k -= 1
                dist[k] = newdist
                u[k] = int(round(E[k][k] / H[k][k]))
                y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
                step[k] = sgn(y)
            else: # 找到更优点
                bestdist = newdist
                u_hat = u.copy()
                k += 1
                u[k] += step[k]
                y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
                step[k] = -step[k] - sgn(step[k])
        else: # 回溯
            if k == n - 1: return u_hat
            else:
                k += 1
                u[k] += step[k]
                y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
                step[k] = -step[k] - sgn(step[k])

# =============================================================================
# 5. BFS (Pohst/Fixed Radius) - 宽度优先逻辑
# =============================================================================

def decode_bfs_pohst(H, r_rot, C_sq):
    """
    Pohst 策略 (类似 BFS): 固定半径搜索
    """
    n = H.shape[0]
    # BFS 必须有一个硬性边界，否则搜索空间无限
    bestdist = C_sq 
    
    best_u = None
    min_dist = C_sq
    
    # 递归实现层层扫描 (BFS behavior simulation)
    def search_layer(k, current_dist, current_u, E_k):
        nonlocal best_u, min_dist
        
        remain = C_sq - current_dist
        if remain < 0: return

        limit = np.sqrt(remain)
        center = E_k[k] / H[k, k]
        
        # 确定搜索范围
        u_min = int(np.ceil(center - limit / abs(H[k, k])))
        u_max = int(np.floor(center + limit / abs(H[k, k])))
        
        # 自然顺序遍历 (Natural Order)
        for val in range(u_min, u_max + 1):
            this_u = current_u.copy()
            this_u[k] = val
            
            d_inc = (E_k[k] - val * H[k, k])**2
            new_dist = current_dist + d_inc
            
            if new_dist <= C_sq:
                if k == 0:
                    # 只有到底层才更新最优解 (BFS 特性: 此时还不能剪枝上面的层)
                    if new_dist < min_dist:
                        min_dist = new_dist
                        best_u = this_u.copy()
                else:
                    E_next = E_k.copy()
                    E_next[:k] = E_k[:k] - val * H[k, :k]
                    search_layer(k-1, new_dist, this_u, E_next)

    init_u = np.zeros(n, dtype=int)
    search_layer(n-1, 0, init_u, r_rot)
    
    if best_u is None:
        return np.zeros(n, dtype=int) 
    return best_u

# =============================================================================
# 6. 主仿真逻辑
# =============================================================================

def run_a2_evaluation():
    G = get_A2_generator()
    n = 2
    G_inv = np.linalg.inv(G) 
    
    H, Q_rot = preprocess_lattice(G)
    
    # --- 关键修改: 范围更细，样本更多 ---
    # 范围: 0 到 9 dB，步长 0.5
    # 样本: 50万 (A2 跑得快，没问题)
    vnrs_db = np.arange(0, 9.1, 0.5)
    trials = 500000 
    
    ser_dfs = []
    ser_bfs = []
    ser_opt = []
    
    print(f"开始 A2 格对比仿真 (样本数: {trials})...")
    print(f"{'VNR(dB)':<8} | {'DFS(SE)':<10} | {'BFS(Pohst)':<12} | {'Optimal':<10}")
    print("-" * 50)
    
    for vnr in vnrs_db:
        sigma = calculate_sigma(G, vnr)
        
        # BFS 半径: 设为 5*sigma，保证绝不漏掉点 (No Erasure)，从而达到 ML 性能
        radius_sq = (5 * sigma)**2 * n 
        
        err_dfs = 0
        err_bfs = 0
        err_opt = 0
        
        # 批量处理以提高 Python 循环效率? A2 只有 2维，直接循环即可。
        for i in range(trials):
            u_true = np.zeros(n, dtype=int) 
            noise = np.random.normal(0, sigma, n)
            y_rx = noise 
            
            r_rot = y_rx @ Q_rot.T
            
            # DFS
            u_dfs = decode_dfs_se(H, r_rot)
            if np.any(u_dfs != 0): err_dfs += 1
            
            # BFS
            u_bfs = decode_bfs_pohst(H, r_rot, radius_sq)
            if np.any(u_bfs != 0): err_bfs += 1
            
            # Optimal
            u_opt = decode_optimal_a2(G_inv, y_rx)
            if np.any(u_opt != 0): err_opt += 1
            
            # 进度打印
            if i % 100000 == 0 and i > 0:
                print(f"  [VNR {vnr}] {i/trials*100:.0f}%...", end='\r')
            
        ser_dfs.append(err_dfs / trials)
        ser_bfs.append(err_bfs / trials)
        ser_opt.append(err_opt / trials)
        
        print(f"\r{vnr:<8.1f} | {ser_dfs[-1]:<10.2e} | {ser_bfs[-1]:<12.2e} | {ser_opt[-1]:<10.2e}")

    # =============================================================================
    # 绘图 - 强调重合性但视觉可分
    # =============================================================================
    plt.figure(figsize=(10, 7))
    
    # 1. Optimal: 粗灰色实线，作为背景基准
    plt.semilogy(vnrs_db, ser_opt, color='gray', linewidth=6, alpha=0.4, label='Optimal Coset Decoder (Ground Truth)')
    
    # 2. DFS: 蓝色圆点，细线
    plt.semilogy(vnrs_db, ser_dfs, 'bo-', linewidth=1.5, markersize=6, label='DFS (Schnorr-Euchner)')
    
    # 3. BFS: 红色叉号，不连线(或者虚线)，稍微错开一点点或直接叠放
    # 为了看清重合，我们不偏移数据，而是依靠标记形状
    plt.semilogy(vnrs_db, ser_bfs, 'rx', markersize=8, markeredgewidth=2, label='BFS (Pohst Fixed Radius)')
    
    plt.title('A2 Lattice Decoding Performance Comparison (N=500k)')
    plt.xlabel('VNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    # 设置下限，避免 0 导致的绘图问题
    plt.ylim(bottom=1e-6) 
    
    plt.savefig('A2_Comparison_HighRes.png')
    plt.show()
    print("\n仿真完成。图像已保存为 A2_Comparison_HighRes.png")

if __name__ == '__main__':
    run_a2_evaluation()