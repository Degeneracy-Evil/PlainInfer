# base-py

个人 Python 工程开发模板，面向通过 GitHub **Use this template** 创建的项目，而不是待发布的 Python 包。

## 开始使用

需要 Python 3.12 或更高版本，使用 uv 管理 Python、虚拟环境、依赖和锁文件。

```bash
uv sync --dev
uv run pre-commit install
```

## 运行

```bash
uv run python src/main.py
```

`src/main.py` 只是用于展示运行、测试和 CI 冒烟检查的最小示例。基于模板创建具体项目后，可按项目需求替换源码结构。

## 质量检查

```bash
uv run python scripts/check.py
```

该命令依次执行 Ruff 自动修复与格式化、mypy strict 和 pytest，并要求分支覆盖率不低于 90%。它可能修改文件；
如果 pre-commit 自动修复了文件，请检查修改、重新暂存并再次提交。CI 使用同一检查入口并额外运行冒烟测试。

## 从模板开始

1. 修改 `pyproject.toml` 中的项目名和描述。
2. 替换 `src/main.py` 与示例测试。
3. 添加项目依赖并更新 `uv.lock`。
4. 更新本文件中的项目说明。

## 许可证

Apache License 2.0
