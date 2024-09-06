import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoModel

tqdm.pandas()

## When testing multiple samples from the dataset, this function allows for batch formatting of the prompts
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f'### Question: {example['question'][i]}'
        output_texts.append(text)
    return output_texts

## Quantization config for loading lower precision models
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
device_map = {"": "cuda:0"}
access_token = None ## PROVIDE OWN HUGGING FACE ACCESS TOKEN HERE

## Load gsm8k dataset
dataset_name = 'gsm8k'
eval_dataset = load_dataset(dataset_name, name='main', split='test')

## Define model here. This could be either a HF model on the hub, or a locally saved model
#model_name = "meta-llama/Llama-2-7b-hf"
model_name = 'C:/Users/ITS490/SeniorProject/trl/ppo_flant5_heuristic'

## Load model using CausalLM for gpt2 and llama2, or Seq2SeqLM for t5. Use line with quantization_config and token for loading llama2
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=True, quantization_config=bnb_config, token=access_token)
#model.config.pretraining_tp = 1 

## Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, token=access_token)
tokenizer.pad_token = tokenizer.eos_token

## Create pipeline for generating text. Use text2text-generation for t5, or text-generation for gpt2 or llama2
generator = pipeline('text2text-generation', model=model_name, tokenizer=tokenizer, max_new_tokens=256)
#generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=256)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

## For use when testing llama2, change eval_dataset[start:end] to slice out specific parts of the dataset 
# for i, entry in enumerate(formatting_prompts_func(eval_dataset[1:4])):
#     print(eval_dataset[i+1])
#     print('\n')
#     eval_tokens = tokenizer.encode(entry)
#     eval_response = model.generate(eval_tokens, **generation_kwargs)
#     print(tokenizer.decode(eval_response))
#     print('\n\n')

## For use when testing gpt2 or t5, change eval_dataset[start:end] to slice out specific parts of the dataset 
# for i, entry in enumerate(formatting_prompts_func(eval_dataset[1:4])):
#     print(eval_dataset[i+1])
#     print('\n')
#     print(generator(entry))
#     print('\n\n')

## For use when testing a single sample, change eval_dataset[0] to access a specific entry in the dataset
query = eval_dataset[0]
query['question'] = '### Question: ' + query['question'] + ' ### Answer: '
print(query)
print('\n')
print(generator(query['question']))

