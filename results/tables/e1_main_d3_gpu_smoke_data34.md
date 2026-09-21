# E1 D3 深度模型单点 GPU 冒烟 — DATA-34（Connectome Architect run, 2026-09-21）

## 冒烟对象
- 线：`origin/agent/connectome-architect/data-43-merge-landing` @ `6fac031`
  （v1.3 active + v1.4 逐字节冻结带 sidecar + §17 ok-only 守卫 + device 修复
  `randperm(..., generator=generator, device=device)` + 逐条落盘
  `write_record(records[-1], raw_dir)`）
- 服务器：`/root/autodl-tmp/drososense/repo`（该 checkout 的 `deep.py:375` 为
  修正版，`runner.py:425` 含逐条落盘，`runner.py:285` 含 ok-only 守卫；
  `protocol --check` 输出 `freeze OK: 1.3.0 ... sha256 d85640e5…`，D3 = LOSO(62)）

## 结果（1 个最小 GRU run，合成窗口，不走 dataset 管线）
- `deep.GRU(task="classification", seed=0, n_channels=8)`，120×16×8：
  fit+predict 2.6 s，`pred shape (120,)`
- `Expected a 'cpu' device type for generator but found 'cuda'`：**0 条**
- 修正表达式 `torch.randperm(n, generator=gen, device=dev)` 实测落在 **cuda:0**

## nvidia-smi 采样（smoke 期间每 2 s 一次）
| 时刻 | GPU util | 显存 |
|---|---:|---:|
| t0 | 0% | 1 MiB |
| t+2s | 0% | 308 MiB |
| t+4s | 2% | 372 MiB |
| 结束后 | 0% | 1 MiB |

（小数据冒烟窗口短，util 采样粒度 2 s；峰值 2%，run 结束后显存回落 1 MiB。）

## 边界说明
- 本次未重跑 4,960 条 D3 深度单元：合流闸门（DATA-43）仍在独立复核，
  本冒烟仅作 device 修复 + ok-only 守卫在 GPU 上的落地证据（只读，
  未写 raw 记录、未杀任何非本任务进程）。
- 已有成果原样保留未重做：D2/GRU 100 ok（wall 1794.7 s）、
  D3 经典 5 模型 1,315 ok（v1.3 LOSO(62) 折）
