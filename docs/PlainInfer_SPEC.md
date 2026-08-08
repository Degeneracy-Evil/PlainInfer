# PlainInfer 开发规格说明

## 1. 项目定位

**PlainInfer** 是一个从零实现、以学习和理解为目标的朴素 LLM 推理引擎。

它的首个目标模型是 **Qwen2.5-0.5B Base**，但项目不应与 Qwen2.5 强绑定。后续可能继续用于学习和支持 Qwen3.x、Llama 等其他 decoder-only Transformer 模型。

本项目的核心目标不是性能，也不是复刻 vLLM、SGLang、TensorRT-LLM 或 Hugging Face Transformers，而是用尽可能直接、透明、可调试的代码，把现代 decoder-only LLM 的推理过程完整实现一遍。

第一阶段需要真正理解并实现：

- 模型配置读取；
- checkpoint 权重加载；
- token embedding；
- RMSNorm；
- RoPE；
- GQA；
- causal self-attention；
- SwiGLU / MLP；
- residual connection；
- decoder layer；
- 完整模型 forward；
- LM Head；
- prefill；
- KV Cache；
- autoregressive decode；
- greedy decoding；
- 基础 sampling；
- 与 Hugging Face 官方实现进行数值对齐。

PlainInfer 是一个**教学项目和实验平台**，而不是生产级 inference framework。

---

## 2. 第一阶段目标模型

第一阶段只实现：

**Qwen2.5-0.5B Base**

使用已经下载好的 Hugging Face / ModelScope 模型目录，允许读取：

- `config.json`
- `*.safetensors`
- tokenizer 相关文件

不重新训练模型，不重新设计 checkpoint 格式。

第一阶段应优先保证 Qwen2.5-0.5B Base 正确运行，不需要兼容所有 Qwen2.5 型号。

---

## 3. 技术栈

第一阶段使用：

- Python
- PyTorch
- transformers
- safetensors
- tokenizers

其中：

- **PyTorch 用于实现真正的模型计算。**
- `transformers` 可以用于 tokenizer、读取官方模型作为 reference、正确性对比。
- `safetensors` 用于读取 checkpoint。
- 不允许调用 Hugging Face 已经实现好的 Qwen 模型完成 PlainInfer 自身的 forward。

允许使用：

- `torch.Tensor`
- `torch.nn.Module`
- `torch.nn.Parameter`
- `torch.nn.Linear`
- `torch.nn.Embedding`
- 基础 tensor operation
- `torch.matmul`
- `torch.softmax`
- reshape / view / transpose / permute / cat 等基础算子

---

## 4. 明确禁止的实现方式

PlainInfer 的核心目标是理解模型推理，因此核心计算不得隐藏在高级封装中。

第一阶段禁止使用：

- `AutoModelForCausalLM` 作为 PlainInfer 的实际推理实现；
- Hugging Face 的 `Qwen2ForCausalLM.forward()`；
- Hugging Face 内部 Qwen attention / decoder layer 实现；
- `torch.nn.functional.scaled_dot_product_attention`；
- FlashAttention；
- xFormers；
- vLLM；
- SGLang；
- TensorRT-LLM；
- DeepSpeed inference；
- accelerate 的模型并行功能；
- PagedAttention；
- continuous batching；
- CUDA Graph；
- 自定义 CUDA kernel；
- Triton kernel；
- Tensor Parallel；
- Pipeline Parallel；
- quantization；
- speculative decoding；
- MTP；
- prefix cache。

Hugging Face 官方模型只能作为**reference implementation**，用于比较中间结果与最终 logits。

---

## 5. “从零实现”的边界

“从零”指的是：

> 自己实现 decoder-only Transformer 的推理计算过程，而不是重新实现所有外围生态。

因此：

### 必须自己实现

- RMSNorm；
- Rotary Position Embedding；
- GQA；
- causal self-attention；
- attention mask；
- Q/K/V projection 组织；
- attention score；
- softmax；
- value aggregation；
- output projection；
- SwiGLU；
- decoder layer；
- residual path；
- full model forward；
- KV Cache；
- prefill / decode；
- generation loop；
- greedy decoding；
- 基础 sampling；
- checkpoint 参数映射。

### 可以直接使用已有库

- Hugging Face tokenizer；
- safetensors 文件解析；
- JSON 解析；
- PyTorch tensor 和基础 operator；
- Hugging Face 官方 Qwen2.5 作为 reference model。

Tokenizer 暂时视为外围组件，不要求自行实现 BPE。

---

## 6. Batch、设备和数值范围

第一版只支持：

- `batch_size = 1`

不要提前实现：

- padding；
- heterogeneous sequence length；
- dynamic batching；
- continuous batching。

第一阶段开发和正确性验证优先使用：

- FP32

在 FP32 正确对齐后，可以增加：

- BF16
- CUDA

CPU 和 CUDA 都应尽量能运行，但正确性优先于性能。

不要因为支持多 device / dtype 引入复杂 abstraction。

---

## 7. 代码设计原则

这个项目最重要的目标是**透明和可理解**。

代码应优先满足：

1. 容易阅读；
2. 容易单步调试；
3. 容易观察 tensor shape；
4. 容易和数学公式对应；
5. 容易和 Hugging Face 中间结果对齐。

不要为了“以后可能支持其他模型”而提前设计复杂框架。

尤其禁止一开始就设计：

- backend registry；
- plugin system；
- complicated model registry；
- generic operator abstraction；
- device abstraction hierarchy；
- execution graph；
- scheduler；
- allocator framework；
- provider pattern；
- factory pattern 堆叠；
- 大量基类。

原则是：

> **先把 Qwen2.5 写对，再在真正出现第二种模型架构时抽象公共部分。**

可以有少量自然的结构，例如：

- config；
- model；
- layers；
- cache；
- generation；
- checkpoint；
- tests。

但必须保持简单。

---

## 8. 代码语言和文档风格

### 标识符

所有：

- class name；
- function name；
- variable name；
- module name

使用英文。

### 注释

注释主要使用中文。

专业术语保留英文，例如：

- hidden state
- attention
- KV Cache
- prefill
- decode
- logits
- head dimension

### README

README 使用中文。

### Tensor shape

核心代码必须尽可能标明 tensor shape。

例如：

```python
# hidden_states: [B, S, hidden_size]
# q: [B, num_attention_heads, S, head_dim]
# k: [B, num_key_value_heads, S, head_dim]
# v: [B, num_key_value_heads, S, head_dim]
```

涉及 reshape、transpose、repeat KV head 等操作时，应明确说明变换前后的 shape。

### 注释原则

不要给显而易见的 Python 语句写废话注释。

重点解释：

- 数学意义；
- tensor shape；
- 为什么需要这一步；
- Qwen2.5 特有结构；
- prefill 和 decode 的差别；
- KV Cache 的数据组织方式。

---

# 9. 开发方式

不要一次性生成整个项目。

必须按照 milestone 顺序逐步实现。

每个 milestone 必须：

1. 实现当前阶段代码；
2. 编写对应测试；
3. 实际运行测试；
4. 修复发现的问题；
5. 给出简短阶段总结；
6. 确认当前 milestone 正确后再继续下一阶段。

不要在前一个阶段明显未通过时继续堆积后续功能。

---

# 10. Milestone 0：项目骨架与模型检查

目标：

建立最简单的工程结构，并确认 Qwen2.5-0.5B Base checkpoint 可以正确读取。

建议结构可以类似：

```text
PlainInfer/
├── plaininfer/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── layers.py
│   ├── rope.py
│   ├── cache.py
│   ├── checkpoint.py
│   └── generation.py
├── tests/
├── scripts/
├── README.md
└── pyproject.toml
```

这只是建议，不要求机械照搬。

本阶段需要：

- 读取 `config.json`；
- 输出关键模型参数；
- 检查 checkpoint tensor 名称；
- 输出主要 tensor shape；
- 确认 Qwen2.5-0.5B 的：
  - hidden size；
  - intermediate size；
  - layer number；
  - attention head number；
  - KV head number；
  - head dimension；
  - vocabulary size；
  - RoPE 参数；
  - RMSNorm epsilon。

此阶段不实现模型 forward。

---

# 11. Milestone 1：Checkpoint 加载

实现自己的 checkpoint loader。

目标：

将 Hugging Face Qwen2.5 checkpoint 中的参数显式加载到 PlainInfer 自己定义的 module 中。

要求：

- 使用 safetensors；
- 清楚展示参数名称映射；
- 不允许依赖 Hugging Face model object 来偷偷完成权重加载；
- 检查 missing key；
- 检查 unexpected key；
- 检查 tensor shape。

应该能清楚看见类似：

```text
model.layers.0.self_attn.q_proj.weight
    ->
PlainInfer layer 0 q_proj.weight
```

参数映射应保持简单，不要设计复杂 registry。

---

# 12. Milestone 2：基础组件

逐个实现并测试：

## RMSNorm

明确实现公式。

不要直接调用 Hugging Face RMSNorm。

需要和官方实现数值对齐。

---

## RoPE

自己实现：

- position；
- inverse frequency；
- cos；
- sin；
- rotate_half；
- Q/K rotary embedding。

需要检查：

- shape；
- broadcasting；
- position offset。

后续 KV Cache decode 会依赖正确 position。

---

## GQA Attention

实现 Qwen2.5 的 Grouped Query Attention。

需要清楚体现：

```text
num_attention_heads
num_key_value_heads
num_key_value_groups
```

必须显式实现：

```text
Q @ K^T
     ↓
scale
     ↓
causal mask
     ↓
softmax
     ↓
attention_probs @ V
```

禁止使用：

```python
scaled_dot_product_attention
```

需要清楚展示 KV head 如何扩展/共享给 query heads。

---

## SwiGLU / MLP

显式实现 Qwen2.5 MLP：

```text
gate_proj
up_proj
SiLU
element-wise multiply
down_proj
```

不要隐藏数学过程。

---

# 13. Milestone 3：Decoder Layer

实现完整 Qwen2.5 decoder layer。

清楚表示：

```text
hidden_states
      │
      ├──────────── residual
      │
   RMSNorm
      │
  Attention
      │
      + residual
      │
      ├──────────── residual
      │
   RMSNorm
      │
     MLP
      │
      + residual
```

要求：

- tensor shape 清晰；
- 每一步可以 debug；
- 可以输出中间 hidden states。

---

# 14. Milestone 4：完整模型 Forward

实现：

```text
input_ids
   ↓
Embedding
   ↓
Decoder Layer × N
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
logits
```

此阶段先不使用 KV Cache。

输入完整 token sequence，每次直接执行 full-sequence forward。

要求：

- batch size = 1；
- causal mask 正确；
- position 正确；
- logits shape 正确。

---

# 15. Milestone 5：Hugging Face 数值对齐

这是整个项目中非常重要的阶段。

必须建立 debug / comparison 工具。

使用同一个：

- tokenizer；
- input；
- FP32；
- checkpoint。

比较 PlainInfer 与 Hugging Face 官方模型。

至少比较：

- token embedding；
- 第一层 RMSNorm；
- Q projection；
- K projection；
- V projection；
- RoPE 后的 Q；
- RoPE 后的 K；
- attention output；
- MLP output；
- 每层 decoder hidden state；
- final norm；
- final logits。

输出：

- max absolute error；
- mean absolute error；
- relative error（合适时）。

不要只比较最终生成文本。

最终 logits 对齐才是主要正确性标准。

当出现误差时，应优先定位**第一个开始明显偏离的中间 tensor**，而不是直接修改最终输出。

---

# 16. Milestone 6：无 KV Cache 的朴素 Generation

在已有 full-sequence forward 基础上，实现最简单 autoregressive generation。

流程：

```text
prompt tokens
      ↓
full forward
      ↓
last-token logits
      ↓
select next token
      ↓
append token
      ↓
整个 sequence 再跑一次 forward
      ↓
...
```

首先实现：

- greedy decoding。

确保可以真正输入 prompt 并输出文本。

这一阶段的目的之一，是形成一个后续 KV Cache 版本的正确性 baseline。

---

# 17. Milestone 7：KV Cache

KV Cache 属于 PlainInfer 第一版的一部分。

但是必须在**无 KV Cache 完整 forward 已经正确之后**再实现。

不要从项目第一天就把所有 attention 逻辑和 cache 混在一起。

需要实现每一层的：

```text
K cache
V cache
```

最开始使用最直观的数据结构即可。

例如逻辑上：

```text
[layer][K/V][sequence][head][head_dim]
```

实际 PyTorch shape 根据 attention 实现选择即可，但必须在注释中说明。

---

# 18. Prefill

prompt 首次输入时执行：

```text
prompt
  ↓
all prompt tokens forward
  ↓
每层生成 K/V
  ↓
保存到 KV Cache
  ↓
得到最后 token logits
```

这一步称为：

**prefill**

要求：

- 一次处理整个 prompt；
- cache 中保存所有 prompt token 的 K/V。

---

# 19. Decode

生成新 token 时：

只输入当前新 token。

对于每一层：

```text
new hidden state
      ↓
new Q
new K
new V

new K/V
  ↓
append to KV Cache

new Q
  ↓
attention with all cached K/V
```

必须能够清楚体现：

> decode 阶段不重新计算过去 token 的 K/V。

这一点是 KV Cache 最核心的学习目标。

---

# 20. KV Cache 正确性测试

对同一 token sequence：

比较：

### 方法 A

full-sequence forward。

### 方法 B

prefill + cached decode。

两者最后 token 的 logits 应在合理浮点误差范围内一致。

还需要测试：

- 连续 decode 多个 token；
- position 是否正确递增；
- cache sequence length 是否正确增长；
- 每层 cache shape 是否正确。

---

# 21. Milestone 8：Generation API

在 KV Cache 正确后，整理一个简单 generation API。

目标接口可以类似：

```python
generate(
    prompt,
    max_new_tokens=32,
    temperature=0.0,
    top_k=None,
    top_p=1.0,
)
```

但不要求和 Hugging Face API 完全一致。

实现：

- greedy；
- temperature；
- top-k；
- top-p。

sampling 应保持简单和透明。

不要引入复杂 generation framework。

---

# 22. 第一版完成标准

PlainInfer v0 第一阶段完成时，应该可以做到：

1. 加载 Qwen2.5-0.5B Base；
2. 使用官方 tokenizer；
3. 自己完成 embedding；
4. 自己实现 RMSNorm；
5. 自己实现 RoPE；
6. 自己实现 GQA；
7. 自己实现 causal attention；
8. 自己实现 SwiGLU；
9. 自己实现 decoder layer；
10. 自己实现完整 model forward；
11. 自己实现 LM Head；
12. 和 Hugging Face logits 数值对齐；
13. 完成无 KV Cache generation；
14. 实现 KV Cache；
15. 实现 prefill；
16. 实现 cached autoregressive decode；
17. 验证 cached decode 与 full forward 一致；
18. 支持 greedy；
19. 支持基础 temperature / top-k / top-p sampling；
20. 可以输入字符串并生成文本。

做到这些以后，第一版就完成。

---

# 23. 第一版明确不做的事情

完成上述目标之后停止扩展。

第一版不做：

- PagedAttention；
- page/block allocator；
- prefix caching；
- continuous batching；
- dynamic batching；
- scheduler；
- multi-request serving；
- OpenAI compatible server；
- Tensor Parallel；
- Pipeline Parallel；
- Expert Parallel；
- distributed inference；
- FlashAttention；
- Triton；
- CUDA kernel；
- quantization；
- speculative decoding；
- EAGLE；
- MTP；
- CUDA Graph；
- graph compiler；
- CPU SIMD optimization。

这些属于后续学习阶段。

不要主动提前加入。

---

# 24. 关于未来模型扩展

PlainInfer 后续可能支持：

- Qwen3.x；
- Llama；
- 其他 decoder-only Transformer。

但是**现在不要提前为这些模型建立复杂 abstraction**。

当前原则：

> Qwen2.5 first, abstraction later.

当真正加入第二种模型时，再比较两种架构：

- 哪些组件真正相同；
- 哪些只是参数不同；
- 哪些结构不同。

然后再提取稳定公共接口。

不要根据猜测提前抽象。

---

# 25. 测试要求

建议使用 pytest。

至少包含三类测试。

## 组件级测试

测试：

- RMSNorm；
- RoPE；
- attention；
- GQA；
- MLP；
- cache。

---

## Reference 对齐测试

和 Hugging Face 比较：

- 中间 tensor；
- hidden states；
- logits。

---

## End-to-End 测试

固定 prompt：

- full forward；
- greedy generation；
- KV Cache generation。

测试结果必须可重复。

---

# 26. Agent 工作要求

你是负责实现这个项目的 coding agent。

工作时遵循以下原则：

1. 不要一次性实现所有 milestone。
2. 每次只推进当前 milestone。
3. 开始修改前先阅读现有代码。
4. 不要无必要重构已经正确的代码。
5. 每次修改后运行相关测试。
6. 不要仅凭“代码看起来正确”就认为完成。
7. 如果和 Hugging Face 不一致，优先定位第一个数值分叉点。
8. 不要为了代码优雅而牺牲透明度。
9. 不要提前优化性能。
10. 不要提前增加未来功能。
11. 发现需求歧义时，优先选择最简单、最利于理解 Transformer 推理的实现。
12. 每个 milestone 完成后简短说明：
    - 实现了什么；
    - 关键设计是什么；
    - 如何验证；
    - 当前测试结果；
    - 下一 milestone 是什么。

---

# 27. 最重要的原则

PlainInfer 的价值不在于“又造了一个推理框架”。

它的价值在于：

> 用最少的隐藏逻辑，把一个真实现代 LLM 从 checkpoint 到 token generation 的完整推理过程展开。

因此所有设计决策都应优先服务于：

**Correctness > Transparency > Simplicity > Extensibility > Performance**

第一阶段尤其如此。

如果一种实现运行更快，但明显降低可读性，不采用。

如果一种 abstraction 看起来更工程化，但让 Transformer 数学过程更难追踪，不采用。

如果 Hugging Face 已经有一个方便的高级函数，但调用它会绕过我们真正要学习的计算过程，不采用。

最终目标不是写出最短代码，而是写出一份可以让人顺着代码完整理解：

```text
token
→ embedding
→ Transformer layers
→ logits
→ next token
→ KV Cache
→ decode
```

全过程的朴素推理引擎。
