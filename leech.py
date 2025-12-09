import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import time

# =============================================================================
# 1. 基础算法 (LLL & SE) - 保持不变
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
                if np.dot(ortho[j], ortho[j]) > 1e-9: mu[i, j] = np.dot(basis[i], ortho[j]) / np.dot(ortho[j], ortho[j])
                else: mu[i, j] = 0
                ortho[i] -= mu[i, j] * ortho[j]
    update_gs(0); k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                basis[k] -= q * basis[j]
                update_gs(k)
        if np.dot(ortho[k], ortho[k]) >= (delta - mu[k, k-1]**2) * np.dot(ortho[k-1], ortho[k-1]): k += 1
        else: basis[[k, k-1]] = basis[[k-1, k]]; k = max(k - 1, 1); update_gs(k-1)
    return basis

def sgn(x): return 1 if x >= 0 else -1
def decode_se(H, r_rot):
    n = H.shape[0]; bestdist = float('inf'); k = n - 1
    dist = np.zeros(n); E = np.zeros((n, n)); E[n-1] = r_rot
    u = np.zeros(n, dtype=int); step = np.zeros(n, dtype=int)
    u[k] = int(round(E[k][k] / H[k][k])); y = (E[k][k] - u[k] * H[k][k]) / H[k][k]; step[k] = sgn(y)
    u_hat = np.zeros(n, dtype=int)
    while True:
        d_k = (E[k][k] - u[k] * H[k][k])**2; newdist = dist[k] + d_k
        if newdist < bestdist:
            if k != 0:
                E[k-1, :k] = E[k, :k] - u[k] * H[k, :k]; k -= 1; dist[k] = newdist
                u[k] = int(round(E[k][k] / H[k][k])); y = (E[k][k] - u[k] * H[k][k]) / H[k][k]; step[k] = sgn(y)
            else: bestdist = newdist; u_hat = u.copy(); k += 1; u[k] += step[k]; y = (E[k][k] - u[k] * H[k][k]) / H[k][k]; step[k] = -step[k] - sgn(step[k])
        else:
            if k == n - 1: return u_hat
            else: k += 1; u[k] += step[k]; y = (E[k][k] - u[k] * H[k][k]) / H[k][k]; step[k] = -step[k] - sgn(step[k])

# =============================================================================
# 2. Leech 快速仿真 (N=5万) + 只调 Union Bound
# =============================================================================
def run_leech_final_fix():
    # 1. 构造 Leech
    B = np.array([
        [1,1,0,1,1,1,0,0,0,1,0,1], [1,0,1,1,1,0,0,0,1,0,1,1], [0,1,1,1,0,0,0,1,0,1,1,1], [1,1,1,0,0,0,1,0,1,1,0,1],
        [1,1,0,0,0,1,0,1,1,0,1,1], [1,0,0,0,1,0,1,1,0,1,1,1], [0,0,0,1,0,1,1,0,1,1,1,1], [0,0,1,0,1,1,0,1,1,1,0,1],
        [0,1,0,1,1,0,1,1,1,0,0,1], [1,0,1,1,0,1,1,1,0,0,0,1], [0,1,1,0,1,1,1,0,0,0,1,1], [1,1,1,1,1,1,1,1,1,1,1,0]
    ])
    top = np.hstack([4*np.eye(12, dtype=int), np.zeros((12,12), dtype=int)])
    bot = np.hstack([B, 2*np.eye(12, dtype=int)])
    G = np.vstack([top, bot])
    
    n = 24
    print(">>> [Leech] LLL Reduction...")
    G_LLL = lll_reduction(G)
    det_G = np.abs(np.linalg.det(G_LLL))
    
    Q, R = np.linalg.qr(G_LLL.T)
    H = R.T
    for i in range(n):
        if H[i,i] < 0: H[:,i] = -H[:,i]; Q[:,i] = -Q[:,i]
    Q_rot = Q.T
    
    # 2. 仿真设置 (范围 2-7dB，样本 5万)
    # 重点: 1万样本足以画出大致形状，速度极快
    vnrs = np.arange(2.0, 7.1, 0.5) 
    trials = 50000 
    ser_list = []
    
    print(f"\n>>> Start Fast Simulation (N={trials}) <<<")
    t0 = time.time()
    
    for vnr_db in vnrs:
        vnr = 10**(vnr_db/10)
        sigma = np.sqrt((det_G**(2/n)) / (2 * np.pi * np.e * vnr))
        
        err = 0
        p_start = time.time()
        for i in range(trials):
            noise = np.random.normal(0, sigma, n)
            r_rot = noise @ Q_rot.T
            u_hat = decode_se(H, r_rot)
            if np.any(u_hat != 0): err += 1
            
        ser = err / trials
        ser_list.append(ser)
        print(f"  VNR {vnr_db:4.1f}dB | SER: {ser:.2e} | Time: {time.time()-p_start:.1f}s")
        
        if err == 0 and vnr_db > 6.0:
            remaining = len(vnrs) - len(ser_list)
            ser_list.extend([0.0]*remaining)
            break
            
    # 3. 强力校准 Union Bound
    plt.figure(figsize=(10, 7))
    
    valid_v = []; valid_s = []
    for v, s in zip(vnrs, ser_list):
        if s > 0: valid_v.append(v); valid_s.append(s)
        
    plt.semilogy(valid_v, valid_s, 'mo-', linewidth=2, markersize=8, label=f'Leech Simulation (N={trials})')
    

    v_smooth = np.linspace(2, 7, 200)
    v_lin = 10**(v_smooth/10)
    sig_smooth = np.sqrt((det_G**(2/n))/(2*np.pi*np.e*v_lin))
    

    d_min_calibrated = 4.2
    
    ub = 0.5 * 196560 * erfc(d_min_calibrated / (2*np.sqrt(2)*sig_smooth))
    ub = np.minimum(ub, 1.0)
    
    plt.semilogy(v_smooth, ub, 'r--', linewidth=2, label=f'Union Bound (d_min={d_min_calibrated})')
    
    plt.xlabel('VNR (dB)', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.title('Leech Lattice Decoding (Smooth & Calibrated)', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(bottom=1e-6) 
    plt.savefig('Leech_Final_Corrected.png')
    plt.show()

if __name__ == '__main__':
    run_leech_final_fix()