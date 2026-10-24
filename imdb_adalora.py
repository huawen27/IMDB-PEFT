import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import logging

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import AdaLoraConfig, get_peft_model, TaskType


# 设定随机种子，便于复现实验
SEED = 42
set_seed(SEED)

logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("imdb_deberta_adalora")

train_df = pd.read_csv(
    "/kaggle/working/labeledTrainData.tsv",
    header=0,
    delimiter="\t",
    quoting=3,
)
test_df = pd.read_csv(
    "/kaggle/working/testData.tsv",
    header=0,
    delimiter="\t",
    quoting=3,
)


# AdaLoRA 在每次优化器更新后都需要更新参数预算和掩码。
# Trainer 的 on_optimizer_step 发生在优化器更新后、梯度清零前。
class AdaLoraAllocationCallback(TrainerCallback):
    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        peft_model = model

        # 兼容 DDP 等包装模型
        while hasattr(peft_model, "module"):
            peft_model = peft_model.module

        update_fn = getattr(peft_model, "update_and_allocate", None)

        if update_fn is None and hasattr(peft_model, "base_model"):
            update_fn = getattr(
                peft_model.base_model,
                "update_and_allocate",
                None,
            )

        if update_fn is None:
            raise RuntimeError(
                "当前 PEFT 模型没有 update_and_allocate()；"
                "请检查 PEFT 版本和 AdaLoRA 模型包装。"
            )

        update_fn(state.global_step)

        return control


if __name__ == "__main__":
    # 固定并分层划分，保证训练集和验证集的情感类别比例接近
    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["sentiment"],
    )

    train_dataset = datasets.Dataset.from_dict({
        "label": train_part["sentiment"].astype(int).tolist(),
        "text": train_part["review"].tolist(),
    })
    val_dataset = datasets.Dataset.from_dict({
        "label": val_part["sentiment"].astype(int).tolist(),
        "text": val_part["review"].tolist(),
    })
    test_dataset = datasets.Dataset.from_dict({
        "text": test_df["review"].tolist(),
    })

    model_id = "microsoft/deberta-v2-xxlarge"
    max_length = 128

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],
    )
    tokenized_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    # 冻结基座配合梯度检查点时，确保输入 embedding 可参与梯度计算
    model.enable_input_require_grads()

    # 训练参数集中定义，避免 total_step 和 Trainer 使用不同的 batch 配置
    num_train_epochs = 2
    train_batch_size = 2
    eval_batch_size = 1
    gradient_accumulation_steps = 2

    # 单 GPU 下按实际优化器更新步数计算
    batches_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    updates_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    total_step = updates_per_epoch * num_train_epochs

    logger.info("训练样本数: %d", len(train_dataset))
    logger.info("每个 epoch 的优化器更新步数: %d", updates_per_epoch)
    logger.info("AdaLoRA total_step: %d", total_step)




    adalora_config = AdaLoraConfig(
        init_r=16,  # AdaLoRA 初始秩；不要再写 r=16
        target_r=8,
        target_modules=["query_proj", "key_proj", "value_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier", "pooler"],
        tinit=100,
        tfinal=500,
        deltaT=10,
        total_step=total_step,
    )
   

    model = get_peft_model(model, adalora_config)
    model.print_trainable_parameters()

    
    # 只转换可训练参数；基座仍保持 FP16，以控制显存
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()

    trainable_dtypes = {
        param.dtype
        for param in model.parameters()
        if param.requires_grad
    }
    print("可训练参数的数据类型:", trainable_dtypes)

    assert torch.float16 not in trainable_dtypes, (
        "仍有 FP16 可训练参数，GradScaler 可能继续报错"
    )
    # 确认分类头和 pooler 确实处于可训练状态
    logger.info("可训练的分类头/pooler 参数：")
    for name, param in model.named_parameters():
        if param.requires_grad and (
            "classifier" in name or "pooler" in name
        ):
            logger.info("%s | dtype=%s", name, param.dtype)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = np.argmax(logits, axis=-1)
        return metric.compute(
            predictions=predictions,
            references=labels,
        )

    training_args = TrainingArguments(
        output_dir="./checkpoint",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # 学习率
        learning_rate=2e-4,

        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        eval_strategy="epoch",

        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        seed=SEED,
        #data_seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[AdaLoraAllocationCallback()],
    )

    trainer.train()

    # 检查验证集指标和预测类别分布
    val_output = trainer.predict(tokenized_val)
    val_logits = val_output.predictions

    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]

    val_pred = np.argmax(val_logits, axis=-1)

    print(
        "验证集预测类别分布:",
        np.unique(val_pred, return_counts=True),
    )
    print(
        "验证集真实类别分布:",
        np.unique(val_output.label_ids, return_counts=True),
    )

    # 预测测试集
    test_output = trainer.predict(tokenized_test)
    test_logits = test_output.predictions

    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]

    test_pred = np.argmax(test_logits, axis=-1).flatten()

    os.makedirs("./result", exist_ok=True)

    result_output = pd.DataFrame({
        "id": test_df["id"],
        "sentiment": test_pred,
    })
    result_output.to_csv(
        "./result/deberta_adalora.csv",
        index=False,
        quoting=3,
    )

    logger.info("结果已保存到 ./result/deberta_adalora.csv")
