# M4 收尾 — 独立统计审查与证据审计 (DATA-5)

**Phase 标注：这是 M4 的收尾段（M4 CLOSING）—— 基于四批已交付证据的
主对比统计、Gate A/B/C、N1–N5 与证据审计表。它不是对 E3/E4/E5/E7 的
宣称；那些批次不存在，本文所有依赖它们的项全部如实标 PENDING。**

审查者：Experimental Statistician（独立统计执行者）。输入全部为已提交、
已验收的 raw/tables 证据；本轮**未跑任何新评测**（D3/E2 raw 只读）。
所有数字可由文末命令清单逐条重建。

---

## 1. E2-D3 汇总核对（Mika 12:20 必办第 1 条）

`results/tables/e2_main_d3_summary.csv` 已由 `scripts/summarize.py` 从
`results/raw/e2_main_d3/**`（8,680 条 ok 记录）重跑聚合，核对通过：

- **14 行**（7 族 × 2 task）＋表头 = 15 行；
- 每族 `n_runs = 620`（10 seeds × 62 folds）＝ 620/族，全批 **8,680 / 8,680，0 失败**；
- `n_distinct_specimens = 62`（LOSO(62)）、`n_auroc_defined = 60/62`（两类齐全折）；
- 每行 `protocol_version=1.4.0 / protocol_compliant=True / status=ok`；
- 独立复算（本审查，从 `e2_main_d3_per_run.csv` 直算 macro_f1/mae 均值）与
  summary **逐格一致**；`e2_main_d2`（700/700）、`e1_main_d3`（11,160/11,160）、
  `e1_main_d2`（900/900）同样核对。
- §17：`e2_main_d3_test_touched_once.json` = 8,680 distinct / **0 violations**。

服务器上的"2 行偏早产物"（`R6/classification`、`R6/regression`，
`n_seeds=3 / n_runs=139`）与已提交真表的差异已确认：**交付一律用重跑后
的本地真表**；`summarize.py` 单写入者后写覆盖缺陷记 DATA-57（backlog，
不在本轮范围），本轮未修改 `summarize.py`。

## 2. 冻结与完整性基准（Gate A 输入）

- 冻结链：`python -m drososense.utils.protocol --check` →
  **freeze OK**（v1.3.0 frozen 2026-09-21T00:45:00Z，sha256 `d85640e556db…`；
  v1.4 sidecar 匹配）。live 首测 2026-09-20，**早于**冻结时间戳（v1.3 是
  R0 返工后的重新冻结；live log 24 条目属 v1.3 前历史，非违规）。
- 全部 4 个 bundle 的 `n_violations = 0`：

| bundle | records | §17 violations | config_hashes（批次作用域） |
|---|---|---|---|
| `e1_main_d2` | 900 | 0 | 4 |
| `e1_main_d3` | 11,160 | 0 | 12 |
| `e2_main_d2` | 700 | 0 | 2 |
| `e2_main_d3` | 8,680 | 0 | 16 |

  `config_hash` 是 **invocation 级**（DATA-48 口径：4 次 fan-out 各一个
  hash；E2-D3 为 8 个分片 × 2 task = 16 个 hash）。**不写"全批共享同一
  配置"，只写"每批各自共享 invocation 配置"。**

## 3. 归一化口径登记（逐字，Mika 12:20 必办第 2 条）

> 正式 E2 批次（`e2_main_d2` / `e2_main_d3`，全部 9,380 条记录）实际使用的
> 归一化为 `model_description.connectome.normalization = "n1_pre_l1"`
> （`npz_path = /root/autodl-tmp/drososense/data/connectome/olfactory_v1.npz`，
> `npz_sha256 = ae86cbb9…`）。11:40Z 裁定原定"正式批次必须使用 `n0_raw`"，
> 其后（DATA-54 落地前）`n0_raw` 的 NPZ-key 映射在活线上不可加载
> （DATA-52 抓到的静默回归同类），`n1_pre_l1` 是**被许可（`ALLOWED_NORMALIZATIONS`
> 之成员）且可加载**的归一化；E2 交付时点选用了 `n1_pre_l1` 并在记录中
> 逐条留痕（`npz_sha256`）。DATA-54 落地后 `n0_raw` 映射已恢复
> （`test_n0_raw_*` 绿），**但已 ok 的单位不重跑**（§17 ok-only、
> DATA-42 口径），`n0_raw` 批次若需作正式对照，属**新批次**，超出本轮
> 范围。
>
> 与 7 个探索性 smoke 单位的 §17 关系：`e2_smoke`（n0_raw）与
> `e2_smoke_rerun`（n5_binary，已裁 void）均在**归档树**
> （`repo_premerge_*` / `archived_raw/` / `repo_retry`），**不进入 live
> `results/raw/` 的 prior-touches 映射**；且其指纹含 model 字段、与正式
> 批次模型指纹不同，**不污染** `e2_main_d2/d3` 与 `e1_main_d2` 的
> 0-violation 结论。据此：正式批次按 §17 字面属**首次触碰**（同 config
> 复算语义只对活树既有触碰成立；归档批次不入守卫映射）。

**对 Gate A 的含义**：协议 §5/§17 的归一化许可集内，`n1_pre_l1` 被许可；
本轮**不**把它记为 CRITICAL（未越出冻结许可集、未改协议、记录逐条留痕、
可比性口径在 §7 的 7 个对比中两侧同一归一化）。但它是一项 **HIGH 记录
项**：n0_raw 裁定未兑现 ⇒ 正式对照口径与 11:40Z 裁定不一致，需 Mika
裁示（是否补 n0_raw 新批次；该裁示不影响本轮判定，只影响后续批次）。

## 4. 主对比 R0_vs_R2（协议 §7 主对比、§10/§11 检验、§12 Holm）

决定性检验 = cluster-level（LOSO fold = specimen-disjoint）精确双侧
符号检验；CI = fold-cluster percentile bootstrap（B=10000，seed 20260920，
取自冻结协议）；效应量 classification=rank-biserial、regression=
Hodges–Lehmann（§10 effect_size 按 task 键控）；`r2` 只作 secondary。

### 4.1 D3（`d3_rainbow_trout`，LOSO(62)，10 seeds）

| 对比 | 指标 | Δ（R0−对照） | 95% CI | 决定性 p | Holm p | 效应量 |
|---|---|---|---|---|---|---|
| **R0_vs_R2（主）** | macro_f1 | **−0.000101** | [−0.000687, +0.000489] | 0.1081 | 0.1081（F_primary 单检验，无校正） | rank-biserial 0.0416 |
| **R0_vs_R2（主）** | mae | **−0.000215** | [−0.000793, +0.000367] | 0.7035 | 0.7035 | Hodges-Lehmann 5.22e-5 |
| R0_vs_R4 | macro_f1 | −0.000434 | [−0.001382, +0.000422] | 0.8776 | **1.0** | rank-biserial 0.0035 |
| R0_vs_R3 | macro_f1 | **−0.001178** | [−0.002261, −0.000197] | 0.04356 | **0.1742** | rank-biserial −0.3261 |
| R0_vs_R5 | macro_f1 | −0.000548 | [−0.001507, +0.000330] | 0.8746 | **1.0** | rank-biserial 0.0305 |
| R0_vs_R1 | macro_f1 | +0.000038 | [−0.000430, +0.000567] | 0.6177 | **1.0** | rank-biserial −0.0221 |

- 主对比 **H1 方向不支持**（R0 未高出 R2；Δ 为负、CI 跨 0、p=0.108 > 0.05）。
- **R0_vs_R3（raw-only，非 Holm 后显著）**：校正前 p = 0.04356
  （< 0.05，raw-only）；Holm 校正后 p = 0.1742 ≥ 0.05 ⇒ **不显著**，
  只能作描述性差异。描述层面 R3（随机稀疏）以 0.00118 macro_f1 的幅度
  优于 R0（方向对 R0 不利，但**不构成 R0 的任何优势**，亦不支持
  R3 > R0 的判定级主张；措辞与 §12 H3 行一致）。注意 D3 上
  n_clusters_nonzero=42、最小可达 p=4.5e-13，可达性本身没有问题
  （不像 D2 的 0.0625 floor），但判定一律取校正后 p。
- 其余 topology 对比 Holm 后均 p=1.0（不可显著）。
- TOST 等价（margin macro_f1=0.02）：R0_vs_R2 与 R0_vs_R4 的 CI 均在
  margin 内 ⇒ **equivalence 成立**（R0 与 R2/R4 在 D3 上统计等价）。
- `R0_vs_GRU`：**unpairable**（E1 基线记录以注册 id `gru` 记，E2 储层
  记录以协议 id `R0..R6` 记；协议 model-zoo 只绑定 `gru→GRU`、`esn→R4`。
  管线按冻结的 model-zoo 绑定解析，**未做就地重命名**——如实报告
  unpairable，不填补。H4 与 Gate_A 的 `params(R0) < params(GRU)` 半边
  因此本轮不可机评（见 §5.1 的人工复核补注）。
- D3 四条硬约束全部执行：AUROC 伴随 `n_auroc_defined`（60/62）；
  macro_f1 不做跨数据集量级比较（D2/D3 仅并列、同注明各自类别覆盖）；
  结论全部走折级 cluster-level 配对；config_hash 按批次作用域（§2 表）。

### 4.2 D2（`d2_beef_uncontrolled`，LOSO(5)，10 seeds）

| 对比 | 指标 | Δ | 95% CI | 决定性 p | Holm p |
|---|---|---|---|---|---|
| R0_vs_R2 | macro_f1 | −0.00711 | [−0.02189, +0.01345] | 0.375 | 0.375 |
| R0_vs_R2 | mae | −0.09690 | [−0.28180, +0.10564] | 1.0 | 1.0 |
| R0_vs_R3 | macro_f1 | +0.00872 | [+0.00062, +0.01638] | 0.375 | **1.0** |
| R0_vs_R4 | macro_f1 | +0.00176 | [−0.01623, +0.01897] | 1.0 | **1.0** |

- D2 只有 5 个 cluster ⇒ 精确双侧符号检验 **floor = 2/2^5 = 0.0625 > 0.05，
  任何对比在 D2 都不可达显著**（协议 `reachability_rule` 的预期，属
  underpowered 报告，不是"未发现差异"）。
- Holm（F_secondary_topology，声明序 R3→R4→R5→R1）把 D2 的所有二级对比
  全部压到 p=1.0；本报告**未**在任何地方使用 Holm 前的 p 下结论
  （D2 R0_vs_R3 的原始 p=0.375 仅为描述）。

### 4.3 跨食比较（N4 相关，仅作并列、不做量级断言）

D2 Δmacro_f1(R0−R2) = −0.00711 vs D3 = −0.000101：两个数据集上 R0 都
不高于 R2（D3 近乎恒等），**排序一致（均≤0）**，但按 §4 硬约束 2 与
跨覆盖分布（D2 每折 4 类齐全；D3 固定四类 + 缺席记 0）**不作量级
比较**，只报排序一致性供 N4 的机械判定（§6）。

## 5. Gate 判定（协议 §13）

机评输入：四 bundle 的 `gates.json`（同一冻结表达式逐字求值）＋本节
人工复核补注。**补注只用于机评因数据形状不可达的半边，全部可复现。**

### 5.1 Gate A — "项目可行"：**FAIL**（CONDITIONAL 意义上的机械 FAIL）

| 项 | 判定 | 证据 |
|---|---|---|
| A1：R0 在 D2、D3 上"不显著差于 R4" | ✅ 满足 | D3：CI [−0.00138, +0.00042] **含 0**（`ci_contains_zero`）且 TOST 等价（margin 0.02）；D2：CI [−0.016, +0.019] 含 0。两项 = 2 |
| A2：`params(R0) < params(GRU)` | ✅ 满足（人工复核） | R0  trainable = **1,004**（classification，reservoir 冻结 250 单位 + ridge readout；regression 251，全部 1,240 条记录恒定）；GRU 最大选中配置 = **4,452**（`model_parameters.json`，212 条记录、8 个 distinct 配置取 max，选优规则已记录） |
| **合成** | `2 == 2 and 1004 < 4452` = **TRUE** | 机评器报 UNEVALUABLE 的原因：`params()` 半边读 committed 参数表，而 `R0` 的条目不在表里（E2 批次晚于 M0/M1 参数表生成）——这是**数据形状缺口，不是 A2 不成立**；上列两个数字均直接出自已提交记录，逐条可复算 |

- **Gate A = PASS（带 1 项 HIGH 记录项：§3 的 n1_pre_l1 归一化口径）**。
  若 Mika 裁示"归一化必须 n0_raw 且需新批次"，则 Gate A 降级为
  CONDITIONAL（等 n0_raw 新批次）。本轮按 §17 字面（n1_pre_l1 属许可集、
  两侧同一归一化、记录留痕）记 PASS + HIGH 记录项，不记 CRITICAL。
  **[Mika 裁决 2026-09-22 14:05 CST 后更新，见 §11]：HIGH 记录项已裁为
  「已披露的口径偏离」而非协议违反 —— 该段落前半句（若……则降级
  CONDITIONAL）作为裁决前状态保留，当前裁定结果以 §11.1 为准：
  Gate A = PASS，无附条件、不重跑、不开新协议版本。**
- `R0_vs_GRU` unpairable **不影响** Gate A 的合成（Gate A 不用该对比；
  只用 R4 半边 + params 半边）。

### 5.2 Gate B — "TAFE 投稿可"：**FAIL**

- `sig(R0,R2,macro_f1,D2)`：不可达（D2 floor 0.0625 > 0.05，§4.2）；
- `sig(R0,R2,macro_f1,D3)`：决定性 p = **0.1081 > 0.05，不显著**；
- `sig_any(R0,R4,macro_f1,[D2,D3],[dropout_p0.3, noise_s0.1,
  train10pct, train25pct])`：**PENDING**（E3/E4/E5 条件批次不存在）。

∴ Gate B = **FAIL（as far as evaluable）**。按 §13 `if_failed`：
走叙事调整规则（§6），**不得**维持"连接组贡献"的性能主张。

### 5.3 Gate C — "强论文"：**FAIL**

- `sig(R0,R2,macro_f1,D3)` = False（p=0.108）；
- `sig(R0,R3,macro_f1,D3)` = 原始 p=0.0436 **可达显著**，但属
  F_secondary_topology，**Holm 后 p = 0.1742 ≥ 0.05 ⇒ 不显著**（§12：
  `sig` 的判定一律用校正后 p，本报告未用未校正值）；
- `sig_any(...)` PENDING（E3–E5 缺）。

机评器输出 `Gate_C.result = false`，与本节一致。Gate C = **FAIL**。

## 6. 叙事规则 N1–N5 逐项裁定（协议 §14）

| 规则 | 触发 | 裁定 |
|---|---|---|
| **N1** | `equiv(R0,R2,macro_f1,D3) and sig(R0,R3,macro_f1,D3)` | **不触发**：equiv 成立（TOST 区间包含），但 `sig(R0,R3)` 需 Holm 校正后显著（0.1742），**不显著**。规则未触发——不得借 N1 把"结构而非接线"写成结论 |
| **N2** | `equiv(R0,R4,D3) and sig_any(R0,R4,...,[dropout_p0.3, noise_s0.1])` | **部分**：equiv(R0,R4,D3) 成立；`sig_any` 半边 PENDING（E4/E5 缺）。**若** E4/E5 落地后 `sig_any` 仍不显著，N2 不触发（N2 要求 R0>R4 在扰动下显著） |
| **N3** | `sig(R4,R0,macro_f1,D3)`（R4 显著优于 R0） | **不触发**：R4 vs R0 的 Holm 校正后 p=1.0，不显著；D3 上 R0≈R4（等价） |
| **N4** | 跨食排序一致性 | 机械判定：D2 的 sig 半边**不可达**（floor 0.0625），D3 的
  `sig(R0,R2)`/`sig(R2,R0)` 均 False（p=0.108）⇒ **N4 整体不触发**，
  且**不得宣称跨食泛化**（两个食物上 R0 都未显著优于 R2） |
| **N5** | 数据集不可用 | **不触发**（D2、D3 均可用；D1 在协议中排除而非缺失） |

**因果说明（"证据变了，不是尺子变了"）**：本轮相对 Phase 1 的预期
（H1 主假设：R0 > R2）证据不支持——主对比 D3 上 p=0.108、Δ≈0、
TOST 等价成立。按 §14 与 Gate B 的 `if_failed` 路径：**叙事必须降级为
"N3/N2 型负结果/等价结果报告"**——生物连接组储层与度保持重接线对照
（R2）及匹配规模 ESN（R4）在 D3 上**统计等价**；R3（随机稀疏）相对
R0 存在小幅描述性差异（Δ=−0.00118 macro_f1，校正前 p=0.04356，**raw-only**；
Holm p=0.174，**未达显著**，只能作描述报告，不构成任何判定级主张）。**不得**保留"生物接线有意义"的主张；
可以（且应当）报告为"连接组拓扑在此任务上不产生可检测优势"的
computational-neuroscience 型负结果。

**叙事口径锁定（Mika 判决二，2026-09-22 14:05 CST，全文见 §11.2）**：
- `R0_vs_R2` 写成 **TOST 等价区间（margin 0.02）内的"无差异"**，
  **不得**写成"趋势向好/接近显著"；
- D2 五 cluster 下最小可达 p = 0.0625 > 0.05 ⇒ 写成"**该数据集下
  不可达**"，**不得**写成"未检出"；
- `R0_vs_GRU` 保持"真实缺口、不插补"的写法，不用其他模型替代；
- 论文主张从"生物连接组带来性能优势"转为 **"生物约束的价值在
  效率/稀疏/低训练成本一侧（Gate A 已 PASS），而不是准确率一侧"**
  —— N3 方向；`Gate B/C` 的 FAIL 必须在**摘要**中说清，不得只放在
  limitation。§7 证据审计表相应行已按本口径重写。

## 7. 证据审计表（主张 ↔ 实验 ↔ 统计量 ↔ CI ↔ 是否支持）

| 候选主张 | 实验 | 统计量（D3 主） | CI / p（Holm 后） | 是否支持 |
|---|---|---|---|---|
| H1：R0 > R2（主假设，macro_f1） | E2-D3 | Δ=−0.000101, p=0.108, rank-biserial=0.042；**TOST（margin 0.02）等价成立** | [−0.000687, +0.000489] | **不支持；按 N3 等价/负结果叙事报告为"TOST 等价区间内的无差异"。禁止写成"趋势向好/接近显著"（Mika 判决二，2026-09-22 14:05 CST）** |
| H2：R0 > R4 | E2-D3 | Δ=−0.000434, p_Holm=1.0；TOST 等价成立 | [−0.001382, +0.000422] | **不支持；描述为统计等价** |
| H3：R0 > R3 | E2-D3 | Δ=−0.001178, p_Holm=0.1742 | [−0.002261, −0.000197] | **不支持**（方向相反、未达 Holm 显著；只能作描述 + N3 型负结果叙事） |
| H4：R0 > GRU | — | unpairable（model-zoo 绑定形状缺口，§4.1） | — | **不可评：真实缺口、不插补、不用其他模型替代；[需补充]：补 `gru` 侧 E2 批次或修订 model-zoo 绑定走协议版本** |
| R0 的准确率为 R2 之上（任何方向） | — | 上列所有 R0_vs_R2 行：Δ 均 ≈0 / 含 0 / p ≥ 0.05（Holm 后） | — | **不支持，且禁止以任何措辞（含"趋势向好"）主张；只报等价** |
| 回归侧主对比 R0 vs R2 | E2-D3 | Δmae=−0.000215, p=0.7035；TOST（margin 0.05）等价成立 | [−0.000793, +0.000367] | **不支持；等价（"无差异"，不写趋势）** |
| D2 任意 topology 显著性 | E2-D2 | 全部 floor-limited（0.0625 > 0.05） | 见 §4.2 | **该数据集下不可达**（5 cluster，underpowered，按协议报告；**禁止写成"未检出"**） |
| 扰动/低数据鲁棒性（E3–E5） | — | 批次不存在 | — | PENDING（**待 Mika 提供条件 bundle 输入；ES 不得自启**，Mika 2026-09-22 14:05 CST 起挂起） |
| 效率（E7 / params 半边） | E2 raw 记录 | R0=1,004 trainable vs GRU=4,452 max | — | **params 半边数值成立（§5.1 A2 半边，Gate A PASS 的基础）**；完整效率对照（时延/内存/MACs）未做 ⇒ 只能作 **params 侧描述**，不得写成完整效率结论 |
| R0_vs_R2 的价值主张（判决二口径） | E2-D3 + params 记录 | 准确率侧：等价（上列各行）；效率侧：1,004 < 4,452 | — | **支持（限 params 半边）**：主张必须落在"生物约束的价值在效率/稀疏/低训练成本一侧（Gate A PASS），而非准确率一侧"；Gate B/C FAIL 在摘要中明示 |
| 跨食泛化 | N4 | 排序一致（均≤0）但无显著项 | — | **不得宣称** |

## 8. 复现记录（单一 manifest）

命令（本 worktree，全部 exit 0；产物在 `results/tables/<bundle>/`）：

```bash
python -m drososense.evaluation.evidence_stats --experiment e1_main_d2 --protocol-version 1.3
python -m drososense.evaluation.evidence_stats --experiment e1_main_d3 --protocol-version 1.3
python -m drososense.evaluation.evidence_stats --experiment e2_main_d2 --protocol-version 1.3
python -m drososense.evaluation.evidence_stats --experiment e2_main_d3 --protocol-version 1.3
python scripts/evidence_audit.py --experiment e1_main_d2 --protocol-version 1.3
python scripts/evidence_audit.py --experiment e1_main_d3 --protocol-version 1.3
python scripts/evidence_audit.py --experiment e2_main_d2 --protocol-version 1.3
python scripts/evidence_audit.py --experiment e2_main_d3 --protocol-version 1.3
python -m drososense.utils.protocol --check
```

- 协议文件：v1.3 逐字节冻结（`d85640e556db…`）；v1.4 为 E2 批次运行
  时的 sidecar；统计参数（alpha 0.05、B=10000、seed 20260920、
  percentile CI、TOST margin 0.02/0.05、Holm 家族）**全部读自冻结
  协议文件**，未在本轮任何脚本中重声明。
- 独立复算交叉（本审查）：符号检验 p、Holm 序（§4）、fold-cluster
  bootstrap CI（R0 D3 classification：0.612051/0.644916，与管线
  逐位一致）、summary↔per_run↔raw 三层均值核对，**全部无差异**。
- 测试：`tests/test_protocol_and_manifests.py` 66/66 绿（含
  `test_no_decision_metric_is_r2` 等三守卫）；`r2` 仅出现在
  `secondary_metrics_reported_only`。
- 环境与成本：本地 CPU（pandas/numpy/scipy），0 GPU、0 服务器负载、
  **0 新评测**；每 bundle 管线约 10–90 s（D3 11,160 行最大）。

## 9. 需补充材料（pending，本轮不填补、不编造）

1. **E3/E4/E5 条件批次**（dropout/noise/low-data）⇒ Gate B/C 的
   `sig_any` 半边、N2 的半边；
2. **`gru` 侧绑定**：E1 基线以注册 id 记录 ⇒ `R0_vs_GRU` 与 H4
   unpairable；需（a）补跑 GRU 侧 E2 对照或（b）model-zoo 绑定
   修订走新协议版本 + 独立复核——**不得就地改协议**；
3. ~~n0_raw 口径裁示~~ **已裁（Mika，2026-09-22 14:05 CST，判决一，
   全文见 §11.1）**：`n1_pre_l1` 承认为「已披露的口径偏离」，**不视为
   协议违反、不重跑、不开新协议版本**。本条 pending 状态关闭；
   若日后要出 `n0_raw` 版本，走新协议版本 + 独立复核（用新实验标签）。
4. **E7 效率完整对照**（latency/memory/MACs）：本轮只核 params 半边，
   效率主张不得写成结论；
5. `e2_main_d3` 的 n0_raw **同配置复算**若要作正式对照，须走新批次 +
   §17 语义（归档 smoke 不入 live 映射，§3 已登记）。

## 10. 违规与风险清单（按严重度）

| 级 | 项 | 证据 | 处置 |
|---|---|---|---|
| HIGH → **已裁（Mika 判决一，2026-09-22 14:05 CST）** | 正式 E2 批次用 `n1_pre_l1`，与 11:40Z"必须 n0_raw"裁定不一致（后补"另记理由"授权在前） | §3；raw 记录 `normalization` 字段 | **承认为「已披露的口径偏离」，不视为协议违反；不重跑、不开新协议版本**；依据＝交付时点 `n0_raw` 不可加载 + 逐条 `npz_sha256` 留痕 + 7 个探索性 smoke 单位在归档树不进 live `prior-touches`；全文见 §11.1；论文 limitation 与 §7 表继续显式披露 |
| MEDIUM | `R0_vs_GRU` unpairable（model-zoo 绑定形状：E1 用注册 id，E2 用协议 id） | §4.1 | 如实报告，不填补；H4/params 半边人工复核（§5.1） |
| LOW | D2 floor 0.0625 > 0.05，全部 D2 对比 underpowered | §4.2 | 按协议 reachability 规则报告，非"无差异" |
| LOW | 服务器 E2-D3 summary 曾为 2 行偏早产物 | §1 | 已由重跑真表替代；单写入者缺陷 = DATA-57（backlog） |
| — | 无 CRITICAL：未见主指标/终点/排除标准/检验方法在 test 结果后被改；未见选择性报告（12 行对比全报，含不显著的 R0_vs_R3 反向效应）；未见 test 复用（§17 四 bundle 全 0 violations）；`r2` 未入任何判定指标 | §2, §4, §5 | — |

## 11. Mika 裁决登记（2026-09-22 14:05 CST）

**裁决人：Mika（闸门签署方）；时间：2026-09-22 14:05 CST。** 以下两条
判决对 M4 交付（`1185587c` + PR #18）生效；ES 本轮未重跑任何已 ok 单位、
未改任何协议文件；判决落进本文的方式为：§5.1/§5.2 附注、§6 口径锁定、
§7 重写行、§9 关闭 pending ③、§10 HIGH 行改判。

### 11.1 判决一：`n1_pre_l1` 承认为「已披露的口径偏离」，不视为协议违反

Mika 逐条核过依据后裁定（与 §3 登记事实一致）：

1. 交付时点 `ALLOWED_NORMALIZATIONS` 里 `n0_raw` **不可加载**
   （NPZ 无 `norm_n0_raw_*` 键），`n1_pre_l1` 是**当时被许可且可加载**
   的唯一选择；
2. 每条记录都带 `model_description.connectome.npz_sha256 = ae86cbb9…`
   → 可追溯到具体连接组产物，不是无据选择；
3. 11:40Z 的 `n0_raw` 偏好是为了 §17 的"同 config 复算"语义，而 7 个
   探索性 smoke 单位（`e2_smoke` / `e2_smoke_rerun`）位于归档树、
   **不进 live `prior-touches`**，所以该理由在本次批次上不成立；
4. `n0_raw` 映射现虽已恢复（DATA-52/54 线），但对同一批已 ok 单位
   以新 config 重跑会被 §17 拒绝（runner 对已 ok 单位的不同 config
   会 `raise`）—— 重跑既违协议也无收益。

**处置（即刻生效，替代 §3/§5.1 中"待裁示/降级 CONDITIONAL"的措辞）**：

- E2 批次**维持**以 `n1_pre_l1` 出具的结果；
- 该偏离登记为「**已披露的口径偏离**」（disclosed calibration
  deviation），**不视为协议违反**；
- **不重跑、不开新协议版本**；
- 该偏离必须继续留在论文的 limitation 与证据审计表（§7）里显式
  披露（已写，保持）；
- 若日后要出 `n0_raw` 版本 ⇒ 走**新协议版本 + 独立复核**，用新
  实验标签，不得就地解释；
- **Gate A 最终判定：PASS，无附条件**（§5.1 前半的 CONDITIONAL 分支
  随本判决关闭）。

### 11.2 判决二：叙事按 N3 负结果/等价方向，不得反向包装

Mika 接受本报告判定（**Gate A PASS；Gate B FAIL；Gate C FAIL**），并
按项目首日规则（"预设假设不成立时，按预先定义的判断规则诚实调整
故事，不选择性报告"）锁定叙事口径：

1. `R0_vs_R2`（D3，LOSO(62)，10 seeds；Δmacro_f1 = −0.000101，
   95% CI [−0.000687, +0.000489]，decisive p = 0.108，**TOST margin
   0.02 等价成立**）→ **必须写成 TOST 等价区间内的"无差异"**，
   **不得**写成"趋势向好"或"接近显著"，也不得写成"果蝇连接组更好"；
2. **D2 的 5 个 cluster 下最小可达 p = 0.0625 > 0.05** → 必须如实写
   "**该数据集下不可达**"，不得写"未检出"；
3. `R0_vs_GRU` unpairable（registry-id vs protocol-id 记录形状差异）→
   保持"**真实缺口、不插补**"的写法，不得用别的模型替代；
4. 论文主张必须从"生物连接组带来性能优势"转为 **"生物约束的价值在
   效率/稀疏/低训练成本一侧（Gate A PASS），而不是准确率一侧"**；
   `Gate B / Gate C` 的 FAIL **必须在摘要里说清**，不能只放在
   limitation。

§6 末尾的口径锁定段与 §7 的重写行即本判决的落地；本判决不改变
任何已交付数字，只约束后续写作。
