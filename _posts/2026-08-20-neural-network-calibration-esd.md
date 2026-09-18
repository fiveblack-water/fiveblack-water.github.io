---
layout: post
title: 神经网络概率校准方案综述
date: 2026-08-20 20:00:00 +0800
categories: 学习记录
tags: [概率校准, ESD, 不确定性估计, 深度学习]
excerpt: 以 ESD Expected Squared Difference 为主线，梳理训练后校准、训练阶段校准、模型级不确定性估计和 Conformal Prediction，并给出方法选择与实验建议。
---

> 整理日期：2026 年 8 月 20 日。适用范围：分类模型置信度与概率校准。

## 核心结论

如果目标是让分类模型在 IID 场景下输出可信概率，推荐先使用 NLL + ESD 进行训练，再用独立校准集执行 Temperature Scaling。如果类别间偏差明显且校准集充足，可以进一步比较 Vector Scaling 或 Dirichlet Calibration。

如果目标是获得分布偏移下的稳健不确定性，单独依赖训练后的 Temperature Scaling 不够，还应考虑 Deep Ensembles、训练阶段校准或 Conformal Prediction。

## 一 基准论文与问题定义

ESD 论文研究的是分类神经网络的置信度校准：当模型给出置信度为 `z` 时，长期正确率应接近 `z`。论文关注的重点不是重新设计分类器，而是将校准目标加入训练过程。

ESD 用不同置信度阈值下的累计准确率与累计置信度之差构造平方差，避免传统 ECE 对离散分箱的依赖。核心论文是 [ESD Expected Squared Difference as a Tuning-Free Trainable Calibration Measure](https://arxiv.org/abs/2303.02472)。

主要特点如下：

- 将 NLL 与 ESD 联合优化，并通过 interleaved training 减少校准损失在训练集上的过拟合。
- 主要比较对象包括 MMCE、SB-ECE，以及训练后的 Temperature Scaling 和 Vector Scaling。
- ESD 没有 MMCE 的核函数超参数或 SB-ECE 的软分箱参数，论文实验显示其在不同 batch size 下相对稳定。
- 总损失中的权重 `λ` 仍然需要选择。论文中的 tuning-free 主要指 ESD 内部不再引入额外校准超参数。

## 二 训练后校准方案

训练后校准不改变主模型参数，而是在独立 calibration set 上学习一个概率映射。这类方法通常成本低、部署简单，但对数据分布变化比较敏感。

### 常见方法对比

| 方案 | 核心思路与优缺点 | 适用建议 |
| --- | --- | --- |
| Temperature Scaling | 用单一温度 `T` 缩放 logits。参数少、稳定，是深度分类模型的默认基线，但不能表达类别特异性偏差。 | 小校准集、IID 场景首选 |
| Vector / Matrix Scaling | 为每类学习缩放和偏置，或学习完整线性映射。表达能力更强，但参数更多，容易过拟合。 | 类别间偏差明显、数据较充足 |
| Beta / Dirichlet Calibration | Beta 面向二分类，Dirichlet 面向多分类；在 log-probability 空间学习更灵活的映射，并可保留恒等映射。 | 需要比 TS 更强的非对称校准能力 |
| Isotonic / BBQ | Isotonic 学习单调非参数映射；BBQ 对多种分箱方式进行贝叶斯组合。灵活但数据需求较高。 | 二分类或拥有较大 calibration set |
| Spline / KS Calibration | 用累计准确率与累计置信度的差异构造无分箱映射，再用样条函数拟合。ESD 的思想受到其启发。 | 希望使用更灵活的非参数映射 |

相关论文：

- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Beyond Temperature Scaling: Dirichlet Calibration](https://arxiv.org/abs/1910.12656)
- [Calibration of Neural Networks using Splines](https://arxiv.org/abs/2006.12800)

## 三 训练阶段校准方案

训练阶段方案通过修改损失函数、正则化项或数据构造方式，让模型在训练过程中形成更合适的置信度。相比后处理，它更有机会改善分布偏移下的表现，但通常会引入额外计算或超参数。

### 常见方法对比

| 方案 | 核心思路与优缺点 | 适用建议 |
| --- | --- | --- |
| Confidence Penalty / Label Smoothing | 惩罚低熵输出或将硬标签替换为软标签，直接抑制过度自信。实现简单，但可能压低本来就应该高置信度的正确预测。 | 快速基线；需要关注 refinement 损失 |
| Mixup | 同时混合输入和标签，兼具数据增强与标签平滑效果，通常能改善 IID 及部分 OOD 校准。 | 视觉任务常用；会改变训练分布 |
| Focal Loss | 降低容易样本的损失权重，使模型不再过度追逐极高置信度。需要选择或自动确定 focal 参数。 | 希望兼顾准确率与校准 |
| MMCE | 用核均值嵌入度量置信度与正确性的分布差异，作为 NLL 的正则项。较能保留正确的高置信度样本，但存在核和权重选择问题。 | 希望直接优化校准且保留高置信度 |
| Soft ECE / SB-ECE | 将离散 ECE 分箱替换为可微软分箱，直接对 ECE 代理进行训练。效果较强，但对软化参数、分箱设置和损失权重敏感。 | 需要直接优化 ECE 的实验场景 |
| ESD | 比较累计准确率与累计置信度的平方差；无显式分箱，提供无偏且一致的估计量，适合与 NLL 联合训练。 | 大模型、希望减少校准超参搜索 |
| Meta-Regularization / Gamma-Net | 为每个样本学习 Focal Loss 参数，并用平滑 ECE 代理优化 meta learner。表达力强，但训练复杂度更高。 | 研究型方案或需要样本级自适应 |

相关论文：

- [Trainable Calibration Measures from Kernel Mean Embeddings](https://proceedings.mlr.press/v80/kumar18a.html)
- [Soft Calibration Objectives for Neural Networks](https://arxiv.org/abs/2108.00106)
- [Calibrating Deep Neural Networks using Focal Loss](https://papers.nips.cc/paper_files/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html)
- [Towards Unbiased Calibration using Meta-Regularization](https://arxiv.org/abs/2303.15057)

## 四 模型级不确定性估计

Bayesian Neural Network、MC Dropout 和 Deep Ensembles 不属于单一的后处理校准器，而是通过建模参数不确定性或多个模型的预测分布来改善不确定性估计。

它们通常计算成本更高，但在分布偏移场景下比单模型后处理更有潜力。

| 方案 | 核心思路 | 适用建议 |
| --- | --- | --- |
| Bayesian Neural Network | 对模型权重建立概率分布，显式建模参数不确定性；训练和推理较复杂。 | 需要较完整的 Bayesian UQ |
| MC Dropout | 测试时多次随机 dropout 并聚合预测；改造成本低，但需要多次推理。 | 资源有限的近似 Bayesian 方案 |
| Deep Ensembles | 训练多个独立模型并平均预测；通常具有较好的校准和 OOD 鲁棒性，但训练成本是单模型的多倍。 | 安全敏感或分布偏移场景 |

相关论文：

- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [Can You Trust Your Model's Uncertainty? Under Dataset Shift](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

## 五 Conformal Prediction 覆盖率校准

Conformal Prediction 与 ESD 的目标不同。ESD 关注概率值是否与真实正确率匹配；Conformal Prediction 关注预测集合是否以指定概率包含真实标签。

例如，在 95% coverage 下，系统希望长期有 95% 的预测集合包含真实类别。

优势：

- 可以给出具有覆盖率保证的预测集合，适合医疗、风控和安全场景。

限制：

- 覆盖率保证不等价于概率分布已经校准。
- 预测集合可能较大，且保证通常依赖 exchangeability 等条件。

参考论文：[Learning Optimal Conformal Classifiers](https://openreview.net/forum?id=t8O-4LKFVx)。

## 六 方法选择建议

- 普通 IID 多分类：训练后先尝试 Temperature Scaling，这是最稳健、最容易复现的 baseline。
- 希望端到端改善校准：使用 NLL + ESD；训练后再在独立 calibration set 上叠加 TS 或 VS。
- 类别间偏差明显：比较 Vector Scaling 与 Dirichlet Calibration；前者更简单，后者表达能力更强。
- 希望保留正确样本的高置信度：优先比较 MMCE、ESD 和 Focal Loss，不要只使用单纯的熵惩罚。
- 存在分布偏移：不要只依据 IID 验证集上的 ECE 选择方案；加入 corruption/OOD 测试，并考虑 Deep Ensembles 或训练阶段校准。
- 需要严格安全保证：在概率校准之外增加 Conformal Prediction，单独报告 coverage 和平均预测集合大小。

## 七 实验与评价清单

- 数据划分：训练集、独立 calibration set、最终 test set 必须分开；不能用 test set 选择温度或其他校准参数。
- 指标：同时报告 Accuracy、NLL、Brier Score、ECE/SCE、Reliability Diagram；不要只看单一 ECE。
- 场景：至少分别评估 IID、轻度分布偏移、明显 OOD；后处理方法在 IID 上有效，不代表部署后仍然可靠。
- 取舍：校准与准确率、refinement、置信度分布之间存在权衡，应同时观察模型是否丢失了合理的高置信度预测。

## 参考文献总览

- [ESD](https://arxiv.org/abs/2303.02472)
- [Temperature / Vector / Matrix Scaling](https://proceedings.mlr.press/v70/guo17a.html)
- [MMCE](https://proceedings.mlr.press/v80/kumar18a.html)
- [Soft Calibration Objectives](https://arxiv.org/abs/2108.00106)
- [Focal Loss Calibration](https://arxiv.org/abs/2002.09437)
- [Dirichlet Calibration](https://arxiv.org/abs/1910.12656)
- [Spline Calibration](https://arxiv.org/abs/2006.12800)
- [Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [Conformal Classifiers](https://openreview.net/forum?id=t8O-4LKFVx)
