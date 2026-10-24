
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import logging
import torch
import numpy as np
import pandas as pd
import datasets
import evaluate
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

program = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "script"
logger = logging.getLogger(program)
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("Visible GPU count:", torch.cuda.device_count())
print("DataLoader workers: 2")

train_df = pd.read_csv("/kaggle/working/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test_df = pd.read_csv("/kaggle/working/testData.tsv", header=0, delimiter="\t", quoting=3)

class StepTimeCallback(TrainerCallback):
    def __init__(self, warmup_steps=10, measure_steps=100):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.step_start = None
        self.step_times = []
        self.total_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is None:
            return
        elapsed = time.perf_counter() - self.step_start
        step = state.global_step
        if step > self.warmup_steps and len(self.step_times) < self.measure_steps:
            self.step_times.append(elapsed)
        if len(self.step_times) == self.measure_steps:
            avg_time = sum(self.step_times) / len(self.step_times)
            print(f"\nBenchmark: {self.measure_steps} optimizer steps average = {avg_time:.3f} seconds")
            print(f"Estimated time for 1 epoch from optimizer steps: {avg_time * state.max_steps / max(1, args.num_train_epochs):.1f} seconds")
            self.step_times.append(float("inf"))

if __name__ == "__main__":
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["sentiment"]
    )

    train_dataset = datasets.Dataset.from_dict({
        "label": train_df["sentiment"].astype(int).tolist(),
        "text": train_df["review"].tolist()
    })
    val_dataset = datasets.Dataset.from_dict({
        "label": val_df["sentiment"].astype(int).tolist(),
        "text": val_df["review"].tolist()
    })
    test_dataset = datasets.Dataset.from_dict({
        "text": test_df["review"].tolist()
    })

    model_id = "microsoft/deberta-v2-xxlarge"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)
    max_length = 128

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )
    tokenized_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./checkpoint",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=125,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        seed=42,
        data_seed=42
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[StepTimeCallback(warmup_steps=10, measure_steps=100)]
    )

    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.dtype != torch.float16, (
                f"FP16 trainable parameter remains: {name}"
            )

    print("All trainable parameters are FP32 or another non-FP16 dtype.")
    
    
    train_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_start

    print(f"Total training time: {train_elapsed / 60:.2f} minutes")

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()

    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame({
        "id": test_df["id"],
        "sentiment": test_pred
    })
    result_output.to_csv("./result/deberta_lora_workers2.csv", index=False, quoting=3)
    print("Result saved!")
