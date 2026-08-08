# PlainInfer

PlainInfer 是一个从零实现、用于学习现代 decoder-only Transformer 推理过程的朴素 LLM 推理引擎。
第一版目标模型是 Qwen2.5-0.5B Base，设计优先级为：正确性、透明度、简单性、可扩展性、性能。

## 环境

项目支持 Python 3.12–3.14，使用 uv 管理依赖：

```bash
uv sync --dev
uv run pre-commit install
```

## 检查模型

```bash
uv run python src/main.py /path/to/Qwen2.5-0.5B
```

该命令输出模型关键配置，以及 checkpoint 中每个 tensor 的名称和 shape，但不会执行模型 forward。

## Python 生成接口

```python
import torch
from transformers import AutoTokenizer

from plaininfer import PlainInferModel, generate

model_dir = "/path/to/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = PlainInferModel.from_pretrained(model_dir, dtype=torch.float32)

text = generate(
    model,
    tokenizer,
    "Hello, world!",
    max_new_tokens=32,
    temperature=0.0,  # 0 表示 greedy
    top_k=None,
    top_p=1.0,
)
print(text)
```

例如将上述代码保存为 `src/demo.py`，再运行 `uv run python src/demo.py`。模板不是待发布的安装包，若脚本放在
`src/` 之外，请用 `PYTHONPATH=src uv run python your_script.py` 运行。

`generate` 首先对整个 prompt 执行 prefill，随后每步只输入一个新 token，并复用逐层 KV Cache。
`generate_ids` 提供 token ids 层接口，`generate_greedy_full` 则是不使用 KV Cache 的正确性 baseline。

模型会检查 `prompt_tokens + max_new_tokens <= max_position_embeddings`。Qwen2.5-0.5B 配置上限是 32768，
但当前教学实现使用显式 attention score，prefill 的显存复杂度为 `O(sequence²)`，因此“配置允许”不等于任意设备都能跑满 32K。
在 A100 80GB、BF16 下已实际验证 16K prompt 和单次 1024-token cached decode；CPU 或小显存 GPU 的可用长度会明显更短。

## Hugging Face 数值对齐

```bash
PYTHONPATH=src uv run python scripts/compare_hf.py /path/to/Qwen2.5-0.5B --prompt "Hello, world!"
```

该脚本仅把 Hugging Face 模型作为 reference，比较 embedding、第一层关键中间 tensor、每层 hidden state、
final norm 和 logits 的 max/mean absolute error 与 relative error。PlainInfer 自身 forward 不调用 Hugging Face 模型实现。

## 质量检查

```bash
uv run python scripts/check.py
uv run python src/main.py
```

若本地已有真实模型，可启用集成测试：

```bash
PLAININFER_MODEL_DIR=/path/to/Qwen2.5-0.5B uv run pytest
```

完整开发目标和约束见 [docs/PlainInfer_SPEC.md](docs/PlainInfer_SPEC.md)。

## 许可证

Apache License 2.0
