import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import time

# =============================================================================
# 1. 纯 Python LLL 规约
# =============================================================================
def lll_reduction(basis, delta=0.99):
    n = len(basis)
    basis = basis.copy().astype(float)
    ortho = np.zeros_like(basis)
    mu = np.zeros((n, n))

    def update_gs(k_start):
        for i in range(k_start, n):
            ortho[i] = basis[i].copy()
            for j in range(i):
                if np.dot(ortho[j], ortho[j]) > 1e-9:
                    mu[i, j] = np.dot(basis[i], ortho[j]) / np.dot(ortho[j], ortho[j])
                else:
                    mu[i, j] = 0
                ortho[i] -= mu[i, j] * ortho[j]
    
    update_gs(0)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                basis[k] -= q * basis[j]
                update_gs(k)
        norm_k = np.dot(ortho[k], ortho[k])
        norm_km1 = np.dot(ortho[k-1], ortho[k-1])
        if norm_k >= (delta - mu[k, k-1]**2) * norm_km1:
            k += 1
        else:
            basis[[k, k-1]] = basis[[k-1, k]]
            k = max(k - 1, 1)
            update_gs(k-1)
    return basis

# =============================================================================
# 2. SE 球形译码器
# =============================================================================
def sgn(x): 
    return 1 if x >= 0 else -1

def decode_se(H, r_rot):
    n = H.shape[0]
    bestdist = float('inf')
    k = n - 1
    dist = np.zeros(n)
    E = np.zeros((n, n))
    E[n-1] = r_rot
    u = np.zeros(n, dtype=int)
    step = np.zeros(n, dtype=int)
    
    u[k] = int(round(E[k][k] / H[k][k]))
    y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
    step[k] = sgn(y)
    u_hat = np.zeros(n, dtype=int)

    while True:
        d_k = (E[k][k] - u[k] * H[k][k])**2
        newdist = dist[k] + d_k
        if newdist < bestdist:
            if k != 0:
                E[k-1, :k] = E[k, :k] - u[k] * H[k, :k]
                k -= 1
                dist[k] = newdist
                u[k] = int(round(E[k][k] / H[k][k]))
                y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
                step[k] = sgn(y)
            else:
                bestdist = newdist
                u_hat = u.copy()
                k += 1
                u[k] += step[k]
                y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
                step[k] = -step[k] - sgn(step[k])
        else:
            if k == n - 1: return u_hat
            else:
                k += 1
                u[k] += step[k]
                y = (E[k][k] - u[k] * H[k][k]) / H[k][k]
                step[k] = -step[k] - sgn(step[k])

# =============================================================================
# 3. E8 主程序 (优化版)
# =============================================================================
def run_e8_optimized():
    # E8 Generator
    G = np.zeros((8, 8))
    G[0]=[2,0,0,0,0,0,0,0]; G[1]=[-1,1,0,0,0,0,0,0]; G[2]=[0,-1,1,0,0,0,0,0]
    G[3]=[0,0,-1,1,0,0,0,0]; G[4]=[0,0,0,-1,1,0,0,0]; G[5]=[0,0,0,0,-1,1,0,0]
    G[6]=[0,0,0,0,0,-1,1,0]; G[7]=[0.5]*8
    
    n = 8
    print(">>> [E8] 初始化: LLL 规约...")
    scale = 1000.0
    G_LLL = lll_reduction(G * 2 * scale) / (2 * scale)
    
    Q, R = np.linalg.qr(G_LLL.T)
    H = R.T
    for i in range(n):
        if H[i,i] < 0: H[:,i] = -H[:,i]; Q[:,i] = -Q[:,i]
    Q_rot = Q.T
    
    # 优化后的 VNR 范围: 0.0 - 6.0 dB, 步长 0.5
    # 这样能画出更细腻的曲线，且不浪费时间在 0 误码区域
    vnrs = np.arange(0.0, 6.1, 0.5)
    trials = 1000000 # 100万样本
    ser_list = []
    
    print(f"\n>>> 开始 E8 仿真 (优化版) <<<")
    print(f"样本数: {trials}")
    print(f"VNR 范围: {vnrs[0]}dB -> {vnrs[-1]}dB (步长 0.5)")
    t0 = time.time()
    
    for vnr_db in vnrs:
        vnr = 10**(vnr_db/10)
        det_G = np.abs(np.linalg.det(G_LLL))
        sigma = np.sqrt((det_G**(2/n)) / (2 * np.pi * np.e * vnr))
        
        err = 0
        p_start = time.time()
        
        # 批量循环
        for i in range(trials):
            if i % 200000 == 0 and i > 0:
                print(f"  [VNR {vnr_db:4.1f}] 进度: {i/trials*100:.0f}%...", end='\r')
            
            noise = np.random.normal(0, sigma, n)
            r_rot = noise @ Q_rot.T
            u_hat = decode_se(H, r_rot)
            if np.any(u_hat != 0): err += 1
            
        ser = err / trials
        ser_list.append(ser)
        print(f"\r  VNR {vnr_db:4.1f}dB | SER: {ser:.2e} | 错误数: {err:6d} | 耗时: {time.time()-p_start:.1f}s")

        # 智能跳出: 如果已经 0 误码，后面就没必要跑了
        if err == 0 and vnr_db >= 5.5:
            print(f"  >>> VNR {vnr_db}dB 已达零误码，停止后续仿真。")
            remaining_len = len(vnrs) - len(ser_list)
            ser_list.extend([0.0] * remaining_len)
            break

    print(f"总耗时: {(time.time()-t0):.1f}s")
    
    # 绘图
    plt.figure(figsize=(10, 7))
    
    # 过滤掉 0 以避免 log 坐标报错
    valid_vnrs = []
    valid_sers = []
    for v, s in zip(vnrs, ser_list):
        if s > 0:
            valid_vnrs.append(v)
            valid_sers.append(s)
            
    plt.semilogy(valid_vnrs, valid_sers, 'bo-', linewidth=2, label=f'E8 Simulation (N={trials})')
    
    # Union Bound
    v_smooth = np.linspace(0, 6.5, 100)
    v_lin = 10**(v_smooth/10)
    sig_smooth = np.sqrt((1.0**(2/8))/(2*np.pi*np.e*v_lin))
    ub = 0.5 * 240 * erfc(np.sqrt(2)/(2*np.sqrt(2)*sig_smooth))
    
    plt.semilogy(v_smooth, ub, 'r--', linewidth=2, label='Union Bound')
    plt.xlabel('VNR (dB)'); plt.ylabel('SER'); plt.title('E8 Lattice Decoding (Optimized Range)')
    plt.grid(True, which='both', alpha=0.3); plt.legend()
    plt.savefig('E8_Optimized.png')
    plt.show()

if __name__ == '__main__':
    run_e8_optimized()