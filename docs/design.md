# NimbusInfer 设计文档

本文说明引擎的目标、数据流、模块职责与关键算法，重点解释**为什么这样设计**。

## 1. 目标与约束

| 目标 | 约束 |
|---|---|
| 推理引擎的每个环节都从零实现 | 唯一第三方依赖是 NumPy（仅作数值后端） |
| 一套代码支持两代主流架构 | 用 `ModelConfig` 的标志位分派，而非复制两份 `forward` |
| 数值正确性可被证明 | KV Cache 必须与全量重算逐元素一致 |
| 可在 CPU / CI 上跑通 | 模型小、无下载、无 GPU 依赖 |

## 2. 数据流

```
文本
  │  tokenizer.encode（字节级 BPE）
  ▼
token ids ──► Embedding(wte) ──► (+ wpe 或 RoPE)
  │
  ▼  N × DecoderBlock
  │     x = x + Attention(Norm(x))
  │     x = x + MLP(Norm(x))
  │        Attention: Q,K,V 投影 → RoPE → 写入 KV Cache → 因果 softmax → 输出投影
  ▼
final Norm ──► lm_head（可与 wte 权重共享）──► logits
  │  采样（greedy / temp / top-k / top-p / repetition penalty）
  ▼
下一个 token ──► 作为下一轮 decode 的唯一输入（复用 KV Cache）
```

## 3. 模块职责

| 模块 | 职责 | 不负责 |
|---|---|---|
| `config.py` | 架构描述、标志位校验、参数量解析式核算 | 张量运算 |
| `backend.py` | NumPy 数值原语（softmax/gelu/silu/norm/mask/linear） | 模型语义 |
| `tokenizer.py` | 字节级 BPE 训练与 HF 格式读写 | 模型 |
| `model.py` | 解码器前向：RoPE、GQA、SwiGLU、因果注意力 | 采样、I/O |
| `cache.py` | 预分配 KV 存储、就地写入、内存核算 | 注意力计算 |
| `sampling.py` | 采样策略（纯函数） | 模型、KV |
| `weights.py` | safetensors 读写、bfloat16 展宽、HF 权重映射 | 模型前向 |
| `quant.py` | int8/int4 对称量化与半字节打包 | 前向 |
| `engine.py` | prefill/decode 循环、流式、checkpoint I/O、跨轮缓存复用 | 底层算子 |
| `hf.py` | 标准库下载 HF 文件、config 映射、转换成本引擎 checkpoint | 模型前向 |
| `server.py` | 浏览器 Demo：SSE 流式、实时指标、注意力热力图 | 模型内部 |
| `chat.py` | 终端 REPL：多轮共享 KV Cache | 模型内部 |
| `bench.py` | 吞吐、加速比、缓存内存 | — |

## 4. 关键算法

### 4.1 因果多头注意力

对第 `i` 层，输入 `X ∈ R^{S×d}`：

```
Q = X W_q,  K = X W_k,  V = X W_v          # W 均为 (d, ·)，即 x @ W
scores = Q Kᵀ / √head_dim                  # 形状 (heads, S, T)
scores = where(key_pos ≤ query_pos, scores, -1e9)
P = softmax(scores, axis=-1)
out = P V → 转置回 (S, d) → out W_o
```

掩码用 `backend.causal_mask(S, T, offset)` 生成：`offset` 是首个 query 的绝对位置，因此 prefill（`offset=0`）与 decode（`offset=缓存长度`）共用同一段代码。

### 4.2 分组查询注意力（GQA）

`n_kv_heads < n_heads` 时，K/V 头数更少，通过 `np.repeat(..., axis=0)` 广播到 query 头数：

```
group_size = n_heads // n_kv_heads
K = repeat(K, group_size, axis=heads)
```

KV Cache 随 `n_kv_heads` 线性缩小，这就是 GQA 省显存的原因。

### 4.3 旋转位置编码（RoPE）

预计算 `cos/sin` 表，形状 `(max_seq_len, head_dim/2)`，按绝对位置切片后旋转：

```
[x1, x2] = split(x)                       # x: (heads, S, head_dim)
out = concat([x1·cos − x2·sin, x1·sin + x2·cos])
```

采用"半拆分（rotate_half）"形式，与 LLaMA / HuggingFace 实现一致。

### 4.4 KV Cache 的复杂度

| | 每步计算量 | 每步显存访问 |
|---|---|---|
| 无缓存 | O(S²·d)（重算整段前缀） | 重算全部 |
| 有缓存 | O(S·d)（仅新 token） | 只读写新增 K/V |

缓存大小 = `2 × n_layers × n_kv_heads × max_seq_len × head_dim × 4` 字节。`KVCache.bytes` 直接给出该值，`bench` 子命令会打印。

### 4.5 字节级 BPE

1. 把每个"词块"编码成 UTF-8 字节，再通过 GPT-2 的 `bytes_to_unicode` 映射为可打印 unicode 字符序列——因此不存在 `<unk>`。
2. **训练**：统计相邻 pair 频次，反复合并最高频 pair，记录 merge 排名；合并只在词块内进行。
3. **编码**：对词块内字符序列，贪心地先应用排名最低的 merge。
4. **解码**：token 字符串 → 字节 → UTF-8。

训练得到的 `vocab.json` + `merges.txt` 与 HuggingFace 格式兼容，所以同一个 `load()` 既能读自训词表，也能读 GPT-2 词表。

### 4.6 量化

对称 round-to-nearest：

```
scale = max|W| / (2^{bits-1} − 1)
q     = round(W / scale).clip(−2^{bits-1}, 2^{bits-1}−1)
W'    = q · scale
```

- `bits=8`：`q` 直接存 int8。
- `bits=4`：`q ∈ [−8, 7]`，偏置到 `[0,15]` 后**两个一组打包进一个 uint8**（低半字节在前），因此体积约为 int8 的一半。
- `group_size` 给定时按行分组，每组独立求 scale，误差更小。

## 5. 关键设计决策

- **权重统一存 `(in_features, out_features)`**，线性层即 `x @ W`。所以 `_lm_head_weight()` 在权重共享时返回 `wte.T`；HF 的 `_proj`（`(out,in)`）在 `weights.py` 里转置，而 GPT-2 的 `c_attn`（本就是 `(in,out)`）只需按列切成 Q/K/V。
- **过滤掉的 logits 用 `-inf` 而不是 `-1e9`**：`softmax` 会减去最大值，`-inf` 精确得到 0，且 `np.isfinite` 判定清晰，便于测试。
- **KV Cache 预分配**：避免每步重新分配与拷贝，`append` 就地写入游标处，`advance` 只在所有层写完后推进一次。
- **解码器为单序列**：为了把注意力、掩码、缓存三条逻辑讲清楚；批处理/连续批处理是明确的后续方向（见 competition.md）。
- **架构差异用标志位而非子类**：新增变体（滑动窗口、MoE）只需扩展 `config.py` 与 `_block`，不破坏主干。

## 6. 参数命名规范

| 名称 | 形状 | 说明 |
|---|---|---|
| `wte` | (vocab, d) | token embedding |
| `wpe` | (max_seq, d) | 学习式位置编码（仅 GPT-2） |
| `h.{i}.norm1/.norm2.weight[.bias]` | (d,) | 前置归一化 |
| `h.{i}.attn.{q,k,v,o}.weight[.bias]` | (d, ·) | 注意力投影 |
| `h.{i}.mlp.fc/.proj.weight[.bias]` | — | GELU-MLP |
| `h.{i}.mlp.{gate,up,down}.weight` | — | SwiGLU-MLP |
| `norm_f.weight[.bias]` | (d,) | 最终归一化 |
| `lm_head.weight` | (d, vocab) | 未共享时存在 |