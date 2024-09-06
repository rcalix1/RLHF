import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead, set_seed
tqdm.pandas()
## Uncomment if using sentence similarity as reward function
#from sentence_transformers import SentenceTransformer, util

## Define parameters for PPO training
ppo_config = PPOConfig(
    model_name="gpt2",
    query_dataset="gsm8k",
    reward_model="sentence-transformers/all-MiniLM-L6-v2",
    learning_rate=1.41e-05,
    mini_batch_size=32,
    batch_size=32,
    seed=0,
    ppo_epochs=5,
    remove_unused_columns=False
)
set_seed(ppo_config.seed)

## Build dataset for PPO training. Inputs are pre-encoded for use during training
def build_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["question"])
        return sample

    ## load gsm8k with datasets
    ds = load_dataset(config.query_dataset, name='main', split="train")
    #ds = ds.shuffle(seed=42)
    #ds = ds.select(range(500))
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

## Create model. Change to AutoModelForSeq2SeqLMWithValueHead for seq2seq models such as t5
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, torch_dtype=torch.float16)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, torch_dtype=torch.float16)

## Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

## create dataset
dataset = build_dataset(ppo_config)

## Create PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

## Set device for training
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

## Uncomment if using sentence similarity as reward function
# reward_model = SentenceTransformer(ppo_config.reward_model)

# def get_reward(answer, response, question=None):
#     answer = reward_model.encode(answer)
#     response = reward_model.encode(response)
#     reward = util.cos_sim(answer, response)[0][0]
#     if reward > 0.9:
#         return torch.tensor(2)
#     if question is not None:
#         question = reward_model.encode(question)
#         punish = util.cos_sim(question, response)[0][0]
#         if punish > 0.97:
#             return torch.tensor(-2)
#         else:
#             return reward - punish
    
#     return reward

## Define heuristic reward function
def get_reward(answer, response, question=None):
    no_spaces_response = response.replace(' ', '')
    correct_steps = [substr.split('>>')[0].split('=')[0] for substr in answer.split('<<')[1:]]

    reward = torch.tensor(0, dtype=torch.float32)
    for step in correct_steps:
        if step in no_spaces_response:
            reward += 1

    if answer.split('####')[1] in response:
        reward += 1
    
    return reward

generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": 128,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

## Training loop
for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), 'epoch: '):
    ## Iterate through dataset batches
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
    
        ## Get response from LLM
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs)
        batch['response'] = tokenizer.batch_decode(response_tensors)
    
        # Compute reward score
        rewards = [get_reward(a, r, q) for a, r, q in zip(batch['answer'], batch['response'], batch['question'])]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

## Uncomment to save model
#model.save_pretrained('ppo_gpt2_heuristic_original')
#tokenizer.save_pretrained('ppo_gpt2_heuristic_original')