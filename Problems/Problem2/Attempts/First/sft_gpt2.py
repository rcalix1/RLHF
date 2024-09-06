from accelerate import PartialState
import torch
from datasets import load_dataset
from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments
from trl import ModelConfig, SFTTrainer
tqdm.pandas()

## Reformat dataset prompts to the match the style that we want 
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

if __name__ == "__main__":
    ## Training Args
    training_args = TrainingArguments(
        output_dir="sft_gpt2",
        logging_strategy="epoch",
        num_train_epochs=3,
    )
    device_string = PartialState().process_index

    ## Model & Tokenizer
    model_config = ModelConfig(
        model_name_or_path="gpt2",
        torch_dtype="auto",
    )
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={'':device_string},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ## Dataset
    dataset_name = "gsm8k"
    train_dataset = load_dataset(dataset_name, name='main', split="train")
    eval_dataset = load_dataset(dataset_name, name='main', split="test")

    ## Training
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=512,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
    )
    trainer.train()

    ## Uncomment to save model
    #trainer.save_model(training_args.output_dir)
