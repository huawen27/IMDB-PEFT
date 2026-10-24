# IMDB-PEFT
##  一、工作总览 

本周围绕 IMDB 影评情感分类任务，开展了基于 DeBERTa-v2-xxlarge 的PEFT实验，尝试了 LoRA、AdaLoRA、P-Tuning 和 Prefix Tuning 四种方法。

目前，LoRA 和 AdaLoRA 的实验结果相对理想，在kaggle的submission评分都达到了0.93。P-Tuning 和 Prefix Tuning 的测试集准确率接近50%，暂未达到预期，说明这两种方法的当前实现或训练配置还需要进一步排查和调整。

## 二、模型设计

### 2.1 任务与基础模型

本次实验将 IMDB 影评情感分类建模为二分类任务，标签分别表示正面和负面情感。基础模型选用 `microsoft/deberta-v2-xxlarge`，并在其上添加分类头，输出影评的情感类别。

数据集中的训练数据按 8:2 划分为训练集和验证集，并采用分层抽样尽量保持两类样本的比例。训练和评估使用相同的数据处理流程：将影评截断至最多 128 个 Token，并使用动态 Padding；Padding 长度对齐到 8，以适配 GPU 计算。

### 2.2 PEFT方案

#### LoRA

LoRA 在模型指定的线性层中加入低秩适配器，将适配器应用于 `query_proj` 和 `value_proj`，秩 `r=8`，缩放系数 `lora_alpha=16`，Dropout 为 `0.05`，并将分类头纳入训练和保存范围。

#### AdaLoRA

AdaLoRA 在低秩适配的基础上动态调整不同适配器的秩预算，使用初始秩 `init_r=16`、目标秩 `target_r=8`，并将适配器应用于 `query_proj`、`key_proj` 和 `value_proj`。其他主要参数包括 `lora_alpha=32`、Dropout `0.05`，以及 `tinit=100`、`tfinal=500`、`deltaT=10` 等秩调整参数。

为配合动态秩分配，训练流程中加入回调，在优化器更新后调用 AdaLoRA 的参数分配更新方法。分类头和 pooler 也被列入需要训练和保存的模块。

#### P-Tuning 与 Prefix Tuning

P-Tuning 和 Prefix Tuning 作为另外两种参数高效微调方案参与实验。它们通过训练提示相关参数，引导预训练模型适应情感分类任务，同时尽量冻结模型主体参数。

这两种方法的当前实验结果约为 50%，暂未达到预期。因此，后续需要结合实际代码配置，进一步核对提示长度、可训练参数范围、初始化方式及优化器等设置。

### 2.3 模型训练

由于 DeBERTa-v2-xxlarge 参数规模较大，代码使用 FP16 加载模型，并启用梯度检查点以降低显存占用。LoRA 和 AdaLoRA 的训练均设置为 2 个 Epoch（其实在第1个epoch时模型就已经收敛了，浪费了一些时间），并通过梯度累积控制显存需求。

##  三、 实验结果
| 序号 | PEFT方法 | Submission Score |
|---:|---|---:|
| 1 | lora | **0.93808** |
| 2 | adalora | **0.93528** |
| 3 | P-tunning | **0** |
| 4 | prefix | **0** |

