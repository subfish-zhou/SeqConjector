# oeis-short20-moon (N≤20, Moonshine-ready)

## Quick Start

**Training**

```bash
python train_torch.py --out ckpt.pt --steps 20000 --bs 128 --m 200000 --moonshine_prob 0.1 --lambda_exec 0.1 --amp
```

**Inference**

Simple case:

```bash
python main.py beam --A examples/A_nat.json --B examples/B_even.json   --n_in 6 --n_chk 6 --ckpt ckpt.pt --beam 512 --time_limit 10
```

Moonshine case:

```bash
python main.py beam \
  --A examples/monster_A.json --B examples/monster_B.json \
  --ckpt ckpt.pt \
  --n_in 3 --n_chk 3 \
  --moonshine --beam 512 --time_limit 10 \
  --relerr0 0.005 --relerr_step 0.008 --relerr_hi 0.3
```

```bash
python main.py beam \
  --A examples/monster_A.json --B examples/monster_B.json \
  --ckpt ckpt.pt \
  --n_in 6 --n_chk 6 \
  --moonshine --beam 512 --time_limit 10 \
  --relerr0 0.005 --relerr_step 0.008 --relerr_hi 0.3
```


---

一个用于"序列→序列"关系合成的最小完整系统（MVP），聚焦 **N≤20** 的短前缀任务，内置：

* 专用 **DSL**（覆盖 `scale/offset/diff/scan/zip/conv/poly/binom/ibinom/euler` 等常见关系）
* **INSERT1 / INSERT2** 原语（仅在第 1 / 第 2 位插入常数；长度不变）——面向 Moonshine 例子
* **程序必须包含 A**：强制要求 B 必须从 A 变换而来，不允许纯原语组合
* **Moonshine 宽检验**（前 `k` 项严格匹配，尾部对数相对误差放宽，阈值随索引线性增长）
* **EGD**（执行引导束搜）：语法约束 + 宽松/严格双执行器前缀剪枝 + "最小误差兜底"
* **模板 + 求参** 快速通道：`SCAN_ADD A`、`INSERT1 B[1] (SCAN_ADD A)`、`INSERT2 B[2] (SCAN_ADD A)`
* 训练：**交叉熵主损失** + **小权重执行宽损失**（对 moonshine-like 合成样本）
* 运行时间统计：成功时输出 `beam / check / total` 秒数；模板命中可选打印 `tpl`

> 目标：在短序列（5–10 项输入）上发现 A→B 的变换程序 `P`，并在额外校验前缀（最多到 N≤20）上验证泛化。

---

## 目录结构

```
oeis_short20_moon/
├── main.py                 # 命令行：eval/beam
├── train_torch.py          # 训练（CE + moonshine 宽损失）
├── README.md               # 本文
├── examples/               # 简单示例 & moonshine toy
│   ├── A_nat.json          # 1,2,3,...
│   ├── B_even.json         # 2,4,6,...
│   ├── A_squares.json      # n^2
│   ├── B_odds.json         # 1,3,5,...
│   ├── A_triangular_src.json / B_triangular_dst.json
│   ├── monster_A.json      # moonshine toy（长度需与B相近；允许不同）
│   └── monster_B.json
└── oeis/
    ├── __init__.py
    ├── program.py          # AST/Program
    ├── parser.py           # 前缀线性化解析 + 语法签名
    ├── interpreter.py      # 执行器（严格/宽松）+ 抽象时间预算
    ├── checker.py          # 严格检验 & moonshine 宽检验
    ├── beam_egd.py         # 语法约束 + EGD + 误差兜底 + time_limit
    └── torch_model.py      # Transformer 解码器 + 轻特征
```

---

## 安装与环境

* Python 3.9+（3.12 也可）
* PyTorch ≥ 2.0（CUDA 可选）
* 依赖（最小）：

  ```bash
  pip install torch numpy
  ```

> 训练脚本默认使用 `torch.amp.autocast('cuda', ...)`（若 GPU 存在），同时兼容 CPU。

---

## 快速开始

### 1) 严格检验（精确等值）

三角数是自然数前缀和：

```bash
python main.py eval \
  --A examples/A_triangular_src.json \
  --B examples/B_triangular_dst.json \
  --program "SCAN_ADD A" \
  --n_in 8 --n_chk 8
```

### 2) Moonshine 宽检验（前 `k` 项严格 + 尾部按比例放宽）

Moonshine toy 示例（需要插入第 1 项 744）：

```bash
python main.py eval \
  --A examples/monster_A.json \
  --B examples/monster_B.json \
  --program "INSERT1 744 SCAN_ADD A" \
  --n_in 3 --n_chk 3 \
  --moonshine --k_strict 3 --relerr0 0.005 --relerr_step 0.008
```

### 3) 束搜（EGD + 10 秒限时）

推荐先用短前缀（3+3）让模板命中更稳：

```bash
python main.py beam \
  --A examples/monster_A.json --B examples/monster_B.json \
  --n_in 3 --n_chk 3 \
  --moonshine --beam 512 --time_limit 10 \
  --relerr0 0.005 --relerr_step 0.008 --relerr_hi 0.3
```

加载训练权重：

```bash
python main.py beam \
  --A examples/A_nat.json --B examples/B_even.json \
  --n_in 6 --n_chk 6 \
  --ckpt ckpt.pt --beam 512 --time_limit 10
```

> 成功时会打印 `TIME beam=...s check=...s total=...s`。
> 若 moonshine 模板先命中（`PRED(TPL)`），也会可选打印 `tpl` 时间（打开 `cmd_beam` 中模板计时即可）。

---

## 训练

交叉熵（CE）主损失 + moonshine 宽损失（权重默认 0.1，仅对 moonshine-like 合成样本启用）：

```bash
python train_torch.py \
  --out ckpt.pt \
  --steps 20000 --bs 128 --m 200000 \
  --moonshine_prob 0.1 --lambda_exec 0.1 --amp
```

可调建议：

* 更快贴近 moonshine：`--moonshine_prob 0.3`
* 更强训练：`--steps 100000`

训练日志中：

* `loss` 为总损失，`ce` 为交叉熵；困惑度约为 `exp(ce)`。
* 可选打印 `exec`（moonshine 宽损失）。

---

## DSL 速查

**原子**
`A`（输入序列）
常量：`-16..16`（词表，训练采样常量受限；运行时模板可读任意 `B[1]/B[2]`）

**一元/扫描**
`SCALE c`, `OFFSET c`, `MAP_ABS`, `MAP_SGN`, `MAP_MOD m`,
`MAP_TAU/MAP_SIGMA/MAP_PHI/MAP_MU/MAP_OMEGA/MAP_BIGOMEGA`,
`SCAN_ADD`, `SCAN_MUL`,
`DIFF_FWD k`（宽松时尾部填 0 / 严格报未定义）, `DIFF_BACK k`

**二元/多元**
`ZIP op k1 k2`（`op∈{ADD,SUB,MUL,MIN,MAX}`；延迟非负、越界在严格模式报错）
`CONV L w1 ... wL`（L≤5）
`POLY D c0 ... cD`（D≤4，线性组合模板）

**索引/变换**
`REIDX k b`（检查全域索引合法）
`SUBSAMPLE k`（k>0）、`REPEAT k`（k>0）

**标准变换**
`BINOM` / `IBINOM`（二项式/逆二项式变换，行缓存）
`EULER`（欧拉变换，缓存 + 严格时保证整除，否则报错）

**插项/删除原语**
`INSERT1 c`、`INSERT2 c`：仅在第 1/2 位插入常数 `c`，其他右移并丢弃末项，**长度不变**。
`DROP_AT_2`：删除第 2 个元素，其他元素前移，**长度减 1**。

> 用于 Moonshine 例子：`INSERT1 744 SCAN_ADD A`。

---

## 检验（Checker）

### 通用规则

* **程序必须包含 A**：所有程序必须包含输入序列 `A`，确保 B 是从 A 变换而来
  - 不允许如 `PRED_IS_EVEN_N` 这样纯原语自己生成的程序
  - 必须有 `A` 节点作为程序树的叶子节点之一

### 严格检验

* 要求 `n_in + n_chk ≤ min(len(A), len(B))`（总长不等也可）
* 前缀逐项严格相等

### Moonshine 宽检验

* 前 `k_strict` 项严格相等（默认 3）
* 尾部误差：`e_i = |log(B_hat[i]/B[i])|`（或比率）；阈值 `tau0 + tau1 * i`
* 通过则 `reason='moonshine_accept'`

---

## EGD 束搜（关键点）

* **程序必须包含 A**：在评估候选程序时，自动过滤不包含 A 的程序
* **语法约束**：token 只从允许集合扩展（例如 `ZIP` 的两个延迟限制在 `0..16`）
* **前缀执行**：宽松执行用于估误差；**严格执行用于早剪枝**（越界/未定义直接丢）
* **误差函数**：前 `k_strict` 必须刚好对上；尾部用 log 相对误差的 RMSE
* **兜底策略**：若没有"合格"候选，返回 **误差最小** 的可解析前缀，而非退回 `A`
* **时间限制**：`--time_limit`（默认 10s）

---

## 模板 + 求参（Moonshine 优先）

在 `--moonshine` 下先试：

1. `SCAN_ADD A`
2. `INSERT1 B[1] (SCAN_ADD A)`
3. `INSERT2 B[2] (SCAN_ADD A)`

命中则直接返回（可选打印 `tpl` 时间）；未命中则进入束搜。

> 这样 **不依赖词表** 也能使用大常数（744/196884…）。

---

## 常见问题（FAQ / Troubleshooting）

* **`len_mismatch:A=... B=...`**
  已改为允许总长不同；只要 `n_in+n_chk ≤ min(lenA,lenB)` 即可。若仍见此错误，请更新到当前版本。

* **`KeyError: '626'`（训练）**
  训练采样的插入常数超出词表范围。当前代码已将训练常量约束在 `[-8,8]`，推理期大常数通过模板从 `B` 读取。

* **`RuntimeWarning: invalid value encountered in divide`（numpy corr）**
  `cheap_features` 已对零方差/NaN/Inf 做防护；可忽略。

* **`exec_fail:zip_oob`**
  由于候选程序使用了负/过大延迟导致越界。新版 EGD 已在**严格前缀执行**阶段剪枝。

* **`head_strict_fail@i`（moonshine）**
  模板阶段或 EGD 阶段头部严格不通过。Moonshine 例子建议用短前缀（3+3），或放宽尾部阈值（如 `--relerr0 0.02 --relerr_step 0.03 --relerr_hi 0.5`）。

---

## 性能/稳定性建议

* Moonshine 例子：先用 `n_in=3, n_chk=3`，再逐步增加；或放宽尾部阈值
* 束宽 `beam=512+` 通常足够；资源允许可更大
* 训练时提高 `--moonshine_prob`（0.3–0.5）有助于模型更快掌握 INSERT+SCAN 的模式
* 如需更大的整数常量词表，可在 `torch_model.py` 扩展 `INTS`，并重新训练

---

## 变更摘要（相对初版）

* 新增 `INSERT1/INSERT2` 原语（仅第 1/2 位）
* `checker.py`：允许 A/B 总长不同；Moonshine 宽检验
* `interpreter.py`：严格/宽松模式；多操作越界防护；欧拉变换缓存与整除检查
* `beam_egd.py`：

  * 语法约束（限制非法参数空间）
  * 宽松/严格双执行器前缀剪枝
  * “最小误差兜底”避免退回 `A`
  * `time_limit` 限时与成功时间打印
* `main.py`：Moonshine 模板快速通道 + （可选）模板耗时打印
* `train_torch.py`：CE 主损失 + 小权重 Moonshine 宽损失；AMP 兼容；数据合成包含 moonshine-like 尾部偏差
