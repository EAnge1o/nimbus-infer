# NimbusInfer 深度说明

## 一句话定位

> 一个**只用 NumPy**、从零实现的大模型推理引擎：分词、注意力、KV Cache、权重解析、量化全部手写，同时支持 GPT-2 与 LLaMA 两代架构。

## 题目定位

大部分"AI 应用"项目都是调用现成模型 API / 框架。读者无法从一段 `model.generate()` 里判断作者是否理解 Transformer。本项目反过来，把**推理运行时**当作研究对象：注意力、位置编码、缓存、量化这些 AI 基础设施的核心机制全部亲手实现。

与同系列项目的关系：`nimbus-vm` / `nimbus-git` / `nimbus-kv` 构成"从零实现"的系统底座，本项目覆盖其中的 **AI 基础设施**。

## 创新点 / 技术深度

1. **一套代码，两代架构**：`ModelConfig` 用标志位（`norm` / `activation` / `pos_encoding` / `n_kv_heads`）描述架构，GPT-2（LayerNorm+GELU+学习式位置+MHA）与 LLaMA（RMSNorm+SwiGLU+RoPE+GQA）共用同一个 `Transformer.forward`。
2. **KV Cache 数值等价性可证**：测试断言"分块 prefill + 逐步 decode"的 logits 与"整段一次性前向"逐元素一致（`atol=1e-4`）。这是很多教学实现会含糊带过、也最容易被忽略的点。
3. **手写二进制格式解析**：自己解析 safetensors（`<uint64 头长度><JSON 头><原始缓冲>`），并处理 bfloat16 → float32 的位运算展宽；不依赖 `safetensors` 包。
4. **真正的 4-bit 打包**：int4 权重偏置到 `[0,15]` 后每字节装两个（低半字节在前），`effective_bits()` 实测 ≈ 4.0，体积约为 int8 的一半（~7.5x vs float32）。
5. **零框架依赖**：无 PyTorch / Transformers / tokenizers / safetensors，仅 NumPy 作为数值后端。
6. **可演示、可真用**：浏览器可视化 Demo（SSE 流式 + 实时指标 + 注意力热力图）、终端 REPL（跨轮复用 KV Cache）、以及把真实 HuggingFace 权重（如 distilgpt2）转成本引擎格式的 `convert` 命令——三者都只用标准库。

## 可验证性（可现场复现的证据）

```bash
# 1) 82 个单元测试全绿
python -m unittest discover -s tests -t . -v

# 2) 一条命令跑通全流程（训练分词器 → 建模型 → 存盘 → 重载 → 生成 → 基准 → 量化）
python demo.py

# 3) 吞吐与 KV Cache 加速比
python benchmarks/bench.py --preset tiny --tokens 24

# 4) 浏览器可视化 Demo（流式 + 实时指标 + 注意力热力图）
python -m nimbus_infer serve --preset tiny --open

# 5) 终端 REPL（跨轮复用 KV Cache）
python -m nimbus_infer chat --model checkpoints/demo-gpt2
```

关键测试与它证明的命题：

| 测试 | 证明的命题 |
|---|---|
| `test_kv_cache_matches_full_recompute` | 缓存推理 = 全量重算（数值） |
| `test_attention_is_causal` | 改未来 token 不影响过去 logits |
| `test_int4_is_actually_packed` | int4 体积约为 int8 一半、有效位宽 ≈ 4 |
| `test_int8_error_within_half_scale` | 量化误差 ≤ 半个量化步长 |
| `test_roundtrip_multiple_dtypes` | safetensors 读写自洽 |
| `test_llama_projection_transpose` | HF `(out,in)` → 本引擎 `(in,out)` 正确 |
| `test_stream_concatenates_to_generate` | 流式输出与一次性输出一致 |
| `test_captured_attention_is_causal` | 捕获到的注意力严格因果（首个 query 只看首个 key） |
| `test_convert_local_directory_end_to_end` | HF 目录 → nimbus checkpoint → 可生成（离线端到端） |
| `test_server.py` 全套 | 真实启动 HTTP 服务，`/api/stream` 有 token+done，`/api/attention` 返回方阵 |

## 演示脚本（建议顺序）

```bash
python demo.py                         # 全流程，屏幕上逐段讲解
python -m nimbus_infer inspect --model checkpoints/demo-gpt2     # 参数量 / 架构 / 缓存占用
python -m nimbus_infer tokenize --tokenizer checkpoints/demo-gpt2/tokenizer \
    --text "the engine caches keys and values"                   # 展示 BPE 合并结果
python benchmarks/bench.py --preset tiny --tokens 32             # KV Cache 加速比
python -m nimbus_infer serve --preset tiny --open                # 浏览器：流式 + 注意力热力图
python -m nimbus_infer chat --model checkpoints/demo-gpt2        # 终端 REPL，讲缓存跨轮复用
```

若能联网，先跑一次 `convert` 拿到真实权重，演示效果最好：

```bash
python -m nimbus_infer convert --repo distilgpt2 --out checkpoints/distilgpt2
python -m nimbus_infer serve --model checkpoints/distilgpt2 --open
```

## 常见问题与回答

**Q：这算"训练"了一个大模型吗？**
对，本项目是**推理引擎**，权重可以是随机的、也可以是真实开源 weights。演示用的是随机初始化，所以输出是无意义的——这正是重点：全流程没有调用任何模型库。

**Q：KV Cache 为什么能加速？**
无缓存时生成第 t 个 token 要重算前 t 个 token 的 K/V（O(t·d) 每步、总体 O(N²)）；有缓存时历史 K/V 已存好，每步只算新 token 的 K/V 并与缓存做注意力（每步 O(N)）。序列越长收益越大，`bench` 打印的加速比随 `--tokens` 增长而上升。

**Q：GQA 和 MHA 差在哪？**
MHA 每个 query 头有独立 K/V 头；GQA 让多个 query 头共享一个 K/V 头（`n_heads/n_kv_heads = group_size`）。KV Cache 大小与 `n_kv_heads` 成正比，所以 GQA 直接省显存，代价是少量表达力。

**Q：权重共享（tie embeddings）怎么实现的？**
`_lm_head_weight()` 在参数里没有 `lm_head.weight` 时返回 `wte.T`，把输入 embedding 矩阵转置当输出投影用，省 `vocab × d` 个参数。

**Q：int4 真的省一半吗？**
`effective_bits()` 实测约 4.0 位/权重。`size_bytes()` 统计的是**打包后**的字节数，所以 int4 ≈ int8 的一半。误差上界由 `test_int4_error_within_half_scale` 固定为半个量化步长。

**Q：可视化 Demo 为什么不用 Flask / FastAPI？**
本项目的约束是"只依赖 NumPy"。Web 部分用标准库 `http.server` + Server-Sent Events，前端是一个内联 HTML，没有任何构建步骤——这也让"零框架依赖"这个特点在 Demo 层同样成立。

**Q：`convert` 真的不 import transformers 吗？**
是。它用 `urllib` 下载 `config.json` / `model.safetensors` / 词表文件，用本项目的 safetensors 解析器读二进制，再用 `convert_hf_gpt2` / `convert_hf_llama` 做命名映射与转置。测试 `test_convert_local_directory_end_to_end` 用一个离线的 HF 目录证明整条链路可用。

**Q：能不能跑真实的 LLaMA / GPT-2？**
可以，`weights.convert_hf_llama` / `convert_hf_gpt2` 负责命名映射与转置；只要提供 HF 的 safetensors 与对应 `ModelConfig`，`Engine` 即可加载。

## 局限与后续方向

- 目前是**单序列**推理；批处理 / 连续批处理（continuous batching）是下一步。
- 无 GPU 后端；`backend.py` 是唯一接触数组的地方，替换为 CuPy 即可上 GPU。
- 未实现采样缓存（beam search）、投机解码、PagedAttention。
- 分词器的 `regex` 属于可选加速项；未安装时退回标准库近似正则（对英文足够，对部分 CJK 边界略粗）。

## 技术栈

Python 3.10+ / NumPy；`unittest` 测试；GitHub Actions 在 Ubuntu 与 Windows 上跑 3.10 / 3.12 / 3.13。