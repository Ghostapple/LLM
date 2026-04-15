# 大语言模型及其后训练

欢迎来到这个组会讲义仓库。

这里整理了大语言模型基础、模型接入方式，以及后训练中的 SFT、RLHF、DPO 和 PEFT。你可以把它当成一份讲义首页，顺着链接进入各个专题阅读。

## 简介

- 适合作为组会分享后的阅读入口
- 侧重概念梳理、公式理解和实践参考
- 所有讲义、图片和补充材料统一收纳在 `docs/` 下

## 快速入口

1. [什么是大语言模型，如何使用和接入大语言模型](<docs/什么是大语言模型，如何使用和接入大语言模型.md>)
2. [后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）.md>)
   - [重参数化 PEFT（Reparameterization PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/重参数化PEFT（Reparameterization PEFT）.md>)
   - [选择性 PEFT（Selective PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/选择性PEFT（Selective PEFT）.md>)
   - [附加式 PEFT（Additive PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/附加式PEFT (Additive PEFT).md>)
   - [提示式 PEFT（Prompt PEFT）](<docs/后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/提示式PEFT (Prompt PEFT).md>)
3. [训练目标：指令精调（SFT）](<docs/训练目标：指令精调（SFT）.md>)
4. [偏好优化 RLHF 与 DPO](<docs/偏好优化RLHF与DPO.md>)
5. [DeepSeek R1 训练流程](<docs/DeepSeek R1训练流程.md>)

## 推荐的库

- [happy-llm](https://github.com/datawhalechina/happy-llm)：偏原理和实践入门，适合先建立大语言模型的整体认知。
- [self-llm](https://github.com/datawhalechina/self-llm)：偏工程实践和本地部署，适合做微调、推理和环境搭建时参考。

## 目录结构

```text
.
├─ README.md
└─ docs/
   ├─ assets/
   ├─ 什么是大语言模型，如何使用和接入大语言模型.md
   ├─ 训练目标：指令精调（SFT）.md
   ├─ 偏好优化RLHF与DPO.md
   ├─ 后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）.md
   ├─ DeepSeek R1训练流程.md
   ├─ 推荐的库.md
   └─ 后训练参数更新方式：部分参数微调（Parameter-Efficient Fine-Tuning，PEFT）/
```
