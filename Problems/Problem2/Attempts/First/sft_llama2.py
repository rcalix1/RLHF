from accelerate import PartialState
import torch
from datasets import load_dataset
from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig
tqdm.pandas()

## Reformat dataset prompts to the match the style that we want 
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

if __name__ == "__main__":
    ## Quantization config for loading lower precision models
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    ## Training Args
    training_args = TrainingArguments(
        output_dir="sft_llama2",
        logging_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
    )
    device_string = PartialState().process_index
    access_token = None ## PROVIDE OWN HUGGING FACE ACCESS TOKEN HERE

    ## PEFT config for more efficient training. Needed with larger models such as llama2
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ## Model & Tokenizer
    model_kwargs = dict(
        trust_remote_code=True,
        device_map={'':device_string},
        torch_dtype=torch.bfloat16
    )
    model_name = "meta-llama/Llama-2-7b-hf"
    device_map = {'':device_string},

    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, token=access_token, **model_kwargs)
    base_model.config.pretraining_tp = 1 
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

    ## Dataset
    dataset_name = "gsm8k"
    train_dataset = load_dataset(dataset_name, name='main', split="train")
    eval_dataset = load_dataset(dataset_name, name='main', split="test")

    ## Training
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
    )
    trainer.train()

    ## Uncomment to save model
    #trainer.save_model(training_args.output_dir)
