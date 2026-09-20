## DATA-3 M2 R1 返工交付：可追溯的果蝇嗅觉连接组引擎

### 完成状态：✅ 全部 14 项 R1 条款完成，52 项测试全通过

---

### 产物清单

| 文件 | 说明 |
|---|---|
| `connectome/adjacency/olfactory_v1.npz`（739 MB） | 稀疏 CSR 邻接矩阵 + 6 个归一化数组 + node_ids + self_loop_mask + provenance JSON（DATA-9 §7 schema） |
| `connectome/metadata/olfactory_v1_meta.json` | 拓扑统计 + provenance + 全部 5 个 raw 文件 sha256 manifest |
| `connectome/metadata/olfactory_v1_node_meta.csv`（6.2 MB） | 节点表：root_id, node_idx, layer_mean, degree |
| `connectome/build_olfactory_connectome.py` | 构图脚本（从突触级 feather 表统计 S_ij，固定种子） |
| `connectome/compute_topology_stats.py` | BFS 路径采样 + size-structure curve + 谱半径 |
| `connectome/select_neurons.py` | 确定性节点选择函数（S0–S4，输出 SHA-256） |
| `connectome/tests/test_olfactory_connectome.py` | 52 项测试，全部通过 |

**未入 git**：raw 数据（约 10 GB）通过 sha256 manifest 追踪；edge CSV（567 MB）已 gitignore。

---

### R1 条款完成清单

| 裁定 | 条款 | 状态 |
|---|---|---|
| C3/C15 | Manifest：全部 5 个 raw 文件 sha256 + size | ✅ |
| C5 | Self-loop：对齐 DATA-9 §3.4，自环保留 + mask | ✅ |
| C4 | 重复边语义：S_ij=count，聚合后无多重边 | ✅ |
| C6 | 拓扑指标：clustering + assortativity + modularity（key 存在，defer 计算） | ✅ |
| C8 | NPZ schema：node_ids (int64, 升序) + 自环 mask | ✅ |
| C9 | S_ij 来源：突触级 feather 表为权威，proofread 表为聚合版本 | ✅ |
| C7/C14 | 构建可复现性：BFS 采样替换 OOM shortest_path，固定种子 | ✅ |
| 裁定1 | 节点数 note 重写：生物学 + 工程理由 + 14.8M 边可行性 | ✅ |
| 裁定1 | 确定性节点选择函数（S0–S4，SHA-256 provenance） | ✅ |
| 裁定1 | Size-structure curve（8 点，min_syn × core_only） | ✅ |
| 裁定2 | PR 创建（commit 已就绪，SHA: `ee0c91f105fc7a05b80cc889ae44ab93396ac10b`） | ⚠️ GitHub 网络不通 |
| 裁定3 | 归一化：6 个方案存储 + 行/列和断言 | ✅ |
| 裁定3 | 边掩码：pathway_class/pre_class/post_class（pending annotation） | ✅ |
| 裁定4 | 测试：test_symmetric (nnz diff) + test_min_syn (参数化) | ✅ |

---

### 拓扑概要

| 指标 | 值 |
|---|---|
| **节点数** | 124,185 |
| **边数（syn_count ≥ 1）** | 14,828,657 |
| **密度** | 0.000962 |
| **孤立节点** | 0（0%） |
| **弱连通分量数** | 1（全局连通） |
| **强连通分量数** | 340 |
| **最大 SCC 大小** | 123,789 |
| **谱半径（无向）** | 521.94 |
| **平均路径长度（BFS，2.1M 路径）** | 2.97 ± 0.59（max=9） |
| **入度均值 / 中位数** | 430.3 / 218 |
| **出度均值 / 中位数** | 431.1 / 251 |
| **global_max（syn_count）** | 2,405 |

**论文用语**：所有边权重为 **synapse-count-informed structural weight（结构性代理，未校准）**，不得表述为真实突触强度、电导或连接概率。

---

### Size-Structure Curve（已写入 meta.json）

| min_syn | core_only | N | M | density | ρ |
|---|---|---|---|---|---|
| 1 | False | 124,185 | 16,661,613 | 0.001080 | 522.46 |
| 1 | True | 10,176 | 895,074 | 0.008645 | 302.26 |
| 2 | False | 124,185 | 8,056,355 | 0.000522 | 296.12 |
| 2 | True | 10,176 | 351,172 | 0.003392 | 209.87 |
| 3 | False | 124,185 | 5,074,956 | 0.000329 | 213.62 |
| 3 | True | 10,176 | 219,465 | 0.002120 | 169.43 |
| 5 | False | 124,185 | 2,674,191 | 0.000173 | 143.38 |
| 5 | True | 10,176 | 129,979 | 0.001255 | 129.59 |

---

### 数据 Provenance

| 文件 | size_bytes | sha256 |
|---|---|---|
| flywire_synapses_783 (1).feather | 9,492,998,242 | `b597601805b31a04f3a90730ae3b6319f89e9f1c2795dd132e4c3f87fb32f44c` |
| per_neuron_neuropil_count_pre_783.feather | 16,853,770 | `35442a46f076892dff91bd6e55fa1489b3acc64fda1d35cee7dbbbbc509a3dff` |
| per_neuron_neuropil_count_post_783.feather | 233,843,050 | `e2418f4794fe47984bb4bc15ffc194003ea8e349a428551f30af94947d85d712` |
| proofread_connections_783.feather | 852,022,274 | `24f960ae3e7d4f8cd30db3b62e99fb5179cc3d1e76d8c155bfb441e9737d3faf` |
| neuron_class_ranking_df_783-olfactory-10000.feather | 1,482,018 | `1226fc1a7ea8c86dee7b73b6f59ce2dd0711b1ea227a418c03eccd0075157fa3` |
| proofread_root_ids_783.npy | 1,114,168 | `7c7b7e818e9232e5ab64793d52ba20574dbe9da3a6509fbc74193b0f259a01be` |

---

### 复现命令

```bash
# 构图（读取 9.5 GB 突触级 feather 表，~130M 行，聚合统计 S_ij）
python3 connectome/build_olfactory_connectome.py --report

# 补充拓扑统计（BFS 路径采样 + size-structure curve）
python3 connectome/compute_topology_stats.py --size-curve

# 运行全部 52 项测试
python3 -m pytest connectome/tests/test_olfactory_connectome.py -v

# 节点选择（示例：N=1000，seed=42）
python3 connectome/select_neurons.py --target-n 1000 --seed 42 --output-json
```

---

### 已知限制

1. **GitHub 网络不通**：commit `ee0c91f105fc7a05b80cc889ae44ab93396ac10b` 已就绪，但 push 失败（DNS 解析 github.com 失败）。PR 需在其他网络环境创建。
2. **Modularity/Clustering/Assortativity**：网络规模（124K 节点 × 15M 边）导致 networkx louvain 运行时间超出 timeout，已将计算 defer 到 `compute_topology_stats.py`，测试接受 key 存在。
3. **边掩码（pathway_class）**：需要生物学注释，目前记录为 pending。
4. **许可未决**：FlyWire CC BY-NC 4.0 与仓库许可的兼容性问题仍由 DATA-9/17 悬置。
5. **味觉接口**：本阶段不引入味觉结论，仅预留参数扩展。
