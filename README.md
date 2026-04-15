# 📚 大语言模型及其后训练

<div align="center">

**一份面向组会分享与课后复习的 LLM 讲义仓库**

<p>
  <img src="https://img.shields.io/badge/语言-中文-1f77b4?style=flat-square" alt="language" />
  <img src="https://img.shields.io/badge/内容-大语言模型%20%7C%20后训练-orange?style=flat-square" alt="topic" />
  <img src="https://img.shields.io/badge/格式-Markdown-success?style=flat-square" alt="format" />
  <img src="https://img.shields.io/badge/阅读方式-按专题跳转-blueviolet?style=flat-square" alt="reading" />
</p>

</div>

> 🔖 目标很简单：先建立对大语言模型的整体认知，再按专题进入 SFT、RLHF、DPO 和 PEFT。

## ✨ 仓库亮点

- 🧭 按“基础认知 → 后训练方法 → 训练流程”组织内容，适合顺着读
- 🧩 覆盖大语言模型接入、指令精调、偏好优化与参数高效微调
- 🖼 所有讲义、图片、PDF 和 Notebook 统一放在 `docs/` 下，便于查找
- 📦 附带推荐资料，方便继续延伸阅读和做实践参考

## 🚀 快速入口

### 📘 基础认知

- [什么是大语言模型，如何使用和接入大语言模型](docs/什么是大语言模型，如何使用和接入大语言模型.md)

### 🛠 后训练与对齐

- [训练目标：指令精调（SFT）](docs/训练目标：指令精调（SFT）.md)
- [偏好优化 RLHF 与 DPO](docs/偏好优化RLHF与DPO.md)
- [后训练参数更新方式：部分参数微调（PEFT）](docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）.md)
  - [重参数化 PEFT（Reparameterization PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/重参数化PEFT（Reparameterization PEFT）.md>)
  - [选择性 PEFT（Selective PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/选择性PEFT（Selective PEFT）.md>)
  - [附加式 PEFT（Additive PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/附加式PEFT (Additive PEFT).md>)
  - [提示式 PEFT（Prompt PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/提示式PEFT (Prompt PEFT).md>)

### 🧪 训练流程

- [DeepSeek R1 训练流程](docs/DeepSeek%20R1训练流程.md)

## 🧠 推荐阅读顺序

1. 先读基础认知，建立“LLM 是什么、怎么用、怎么接入”的整体框架
2. 再看 SFT 和 PEFT，理解模型如何被任务化和低成本微调
3. 接着进入 RLHF 与 DPO，理解偏好对齐的核心逻辑
4. 最后参考 DeepSeek R1 训练流程，把知识串成完整实践链路

## 📦 推荐的库

- [happy-llm](https://github.com/datawhalechina/happy-llm)：偏原理和实践入门，适合先建立大语言模型的整体认知
- [self-llm](https://github.com/datawhalechina/self-llm)：偏工程实践和本地部署，适合做微调、推理和环境搭建时参考

