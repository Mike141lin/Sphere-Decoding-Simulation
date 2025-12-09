# Sphere Decoding for Lattice Codes: Performance & Complexity Analysis

## 📖 项目简介 (Introduction)

本项目是关于 **球形解码 (Sphere Decoding)** 算法在不同维度格 (Lattice) 下的性能仿真与分析。项目实现了两种主流的解码策略，并深入探讨了高维格的纠错性能及基规约 (Basis Reduction) 的影响。

This repository contains the simulation code and analysis for Sphere Decoding algorithms applied to Lattice Codes. It compares Depth-First Search (DFS) and Breadth-First Search (BFS) strategies across various lattice dimensions (A2, E8, BW16, Leech).

## 🚀 主要功能 (Key Features)

* **多维度格仿真**: 支持 $A_2$ (2D), $E_8$ (8D), $BW_{16}$ (16D) 等典型格结构。
* **算法对比**: 
    * **DFS (Schnorr-Euchner)**: 基于 SE 策略的深度优先搜索。
    * **BFS (Viterbo-Boutros)**: 基于分层搜索的广度优先搜索。
    * 对比指标：符号误码率 (SER)、平均访问节点数 (Visited Nodes)、运行时间 (Runtime)。
* **理论验证**: 将仿真结果与理论 **联合界 (Union Bound)** 进行对比，验证算法的最优性。
* **LLL 规约分析**: 探究 LLL 基规约对标准基 (Good Basis) 和坏基 (Bad Basis) 的解码效率影响。
* **高性能**: 利用 Python `multiprocessing` 实现多核并行仿真，支持百万级 (1,000,000) 样本测试。

## 📂 文件说明 (File Structure)

* `code/A2_Simulation.py`: $A_2$ 格的基本仿真。
* `code/E8_Comparison.py`: $E_8$ 格下 DFS 与 BFS 的效率与性能全方位对比 (节点数、时间、SER)。
* `code/BW16_Simulation.py`: $BW_{16}$ 格 (基于 Reed-Muller 构造) 的高精度仿真与 Union Bound 验证。
* `code/LLL_Analysis.py`: LLL 基规约对解码复杂度的影响分析。
* `Final_Report.tex`: 项目期末报告 (LaTeX 源码)。

## 🛠️ 依赖 (Requirements)

本项目基于 Python 3 开发，依赖以下库：
* numpy
* scipy
* matplotlib
* pandas

安装命令:
```bash
pip install -r requirements.txt
📊 实验结果示例 (Results)1. E8 Lattice: DFS vs BFS在低信噪比区域，DFS 算法的访问节点数比 BFS 少 10-50 倍，展现了极高的搜索效率。2. BW16 Lattice: Simulation vs Theory在 100 万样本的高精度测试下，BW16 的仿真曲线在高信噪比区域完美贴合理论 Union Bound。📝 LicenseMIT License
```text:.gitignore:.gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE settings
.idea/
.vscode/
*.swp

# Simulation Results (Optional - remove if you want to upload images)
# *.png
# *.csv
# *.log

# LaTeX
*.aux
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
*.pdf
```eof

```text:requirements.txt:requirements.txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
```eof
# Sphere-Decoding-Simulation
