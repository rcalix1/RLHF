## RLHF Theoretical Foundations

* Baby RLHF




```


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Tiny Policy Model (like babyGPT, no transformer) ===
class TinyPolicy(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)  # Simple embedding
        self.head = nn.Linear(8, 1)               # Output score for token

    def forward(self, token_ids):
        x = self.embed(token_ids)  # [batch, 8]
        return self.head(x).squeeze(-1)  # [batch] — logits for taken tokens


# === Reward function: +1 if token == 1 ===
def fake_reward(token_ids):
    return torch.where(token_ids == 1, torch.tensor(1.0), torch.tensor(0.0))


# === PPO Loss Function ===
def ppo_loss(new_logits, old_logits, actions, rewards, kl_coeff=0.1):
    new_logprobs = F.logsigmoid(new_logits)
    old_logprobs = F.logsigmoid(old_logits).detach()  # fixed baseline

    # Importance sampling ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # PPO surrogate objective (no advantage, just raw reward)
    surrogate = ratio * rewards

    # KL divergence
    kl = (old_logprobs - new_logprobs).mean()

    # Final PPO loss
    return -surrogate.mean() + kl_coeff * kl


# === Training Loop ===
vocab_size = 10
policy = TinyPolicy(vocab_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for step in range(300):
    # === Sample tokens ===
    logits = torch.randn(vocab_size)  # dummy logits for sampling space
    probs = F.softmax(logits, dim=0)  # [vocab]
    actions = torch.multinomial(probs, num_samples=8, replacement=True)  # [8 tokens]

    # === Old policy output (logits for those tokens) ===
    with torch.no_grad():
        old_logits = policy(actions)  # [8]

    # === Reward based on action taken ===
    rewards = fake_reward(actions)  # [8], +1 if token == 1 else 0

    # === New policy output ===
    new_logits = policy(actions)  # [8]

    # === Compute PPO loss ===
    loss = ppo_loss(new_logits, old_logits, actions, rewards)

    # === Optimize ===
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # === Print progress ===
    if step % 50 == 0:
        avg_reward = rewards.float().mean().item()
        print(f"Step {step} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.2f}")




```

## Other




```



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Tiny GPT-style Policy ===
class TinyPolicy(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)  # Predict next-token logits

    def forward(self, input_ids):
        x = self.embed(input_ids)          # [seq_len, hidden]
        last_hidden = x[-1]                # Use last token's embedding
        logits = self.lm_head(last_hidden) # [vocab_size]
        return logits  # logits over vocab for next token


# === Reward function: +1 if token == 1 ===
def fake_reward(token_ids):
    return torch.where(token_ids == 1, torch.tensor(1.0), torch.tensor(0.0))

# === PPO Loss Function ===
def ppo_loss(new_logprobs, old_logprobs, rewards, kl_coeff=0.1):
    ratio = torch.exp(new_logprobs - old_logprobs)
    surrogate = ratio * rewards
    kl = (old_logprobs - new_logprobs).mean()
    return -surrogate.mean() + kl_coeff * kl


# === Training Loop ===
vocab_size = 10
policy = TinyPolicy(vocab_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for step in range(300):
    prompt = torch.tensor([0])  # Start with a fixed token (e.g., BOS token 0)
    generated = [0]

    old_logprobs = []
    rewards = []

    for _ in range(8):  # Generate 8 tokens step-by-step
        logits = policy(prompt)
        probs = F.softmax(logits, dim=-1)
        # torch.multinomial adds controlled randomness — usually picks high-prob tokens like index 8  
        # but sometimes explores others — perfect for RLHF-style exploration vs argmax
        next_token = torch.multinomial(probs, num_samples=1)  # Sample next token
        logprob = torch.log(probs[next_token])

        # Save data
        old_logprobs.append(logprob)
        rewards.append(fake_reward(next_token))
        generated.append(next_token.item())

        # Update prompt
        prompt = torch.cat([prompt, next_token], dim=0)

    generated_tensor = torch.tensor(generated[1:])  # drop BOS token
    rewards = torch.stack(rewards).squeeze()
    old_logprobs = torch.stack(old_logprobs).squeeze()

    # === Compute new logprobs under current policy ===
    new_logprobs = []
    prompt = torch.tensor([0])
    for token in generated_tensor:
        logits = policy(prompt)
        probs = F.softmax(logits, dim=-1)
        logprob = torch.log(probs[token])
        new_logprobs.append(logprob)
        prompt = torch.cat([prompt, token.unsqueeze(0)], dim=0)

    new_logprobs = torch.stack(new_logprobs)

    # === Compute PPO loss ===
    loss = ppo_loss(new_logprobs, old_logprobs, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        avg_reward = rewards.float().mean().item()
        print(f"Step {step} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.2f}")




```







### 🔢 `probs` vs `logprobs`

| Term       | Meaning                           | Example           |
|------------|------------------------------------|-------------------|
| `probs`    | Regular probabilities (0 to 1)     | `0.2`, `0.7`, `0.05` |
| `logprobs` | Logarithm of those probabilities   | `log(0.2) ≈ -1.61`   |

### 🧠 Why use `logprobs`?

- Avoids tiny numbers (numerically stable)
- Makes math additive: `log(p1 * p2) = log(p1) + log(p2)`
- PPO uses `logprobs` to compute: `log(π_new) - log(π_old)`  
  → this log-ratio controls the policy update size

### 🧪 Tiny PyTorch Example

```python
probs = torch.tensor([0.2, 0.7, 0.1])
logprobs = torch.log(probs)



### 🧠 PPO: Why Use `probs`, `logprobs`, and Ratios?

In PPO (used in RLHF), we train a policy by generating actions (e.g., tokens), scoring them, and updating the model carefully.

To control how much the model changes, PPO compares the new and old policy’s confidence using a ratio:

**ratio = π_new(action) / π_old(action)**

To make this stable, we compute it in log-space:

* `logprobs = log(π(action))`
* `log_ratio = logπ_new - logπ_old`
* `ratio = exp(log_ratio)`

This tells us:

* `ratio > 1.0`: model is more confident now
* `ratio < 1.0`: model is less confident

Then PPO uses this in the loss:

**loss = - ratio * reward + KL_penalty**

This encourages:

* Higher reward sequences to be more likely
* But keeps the policy close to the old one (trust region)

✅ Use `probs` to sample actions
✅ Use `logprobs` to track confidence
✅ Use `logπ_new - logπ_old` to compute the ratio for PPO updates




### 🔢 What’s the "ratio" in PPO?

You're thinking:

> "Ratio = a / b" → ✅ that's exactly what it is.

In PPO, the ratio is:

```
π_new(action) / π_old(action)
```

But since we work with **logprobs**, we use:

```
log(π_new(action)) - log(π_old(action)) = log(π_new / π_old)
```

Then we take:

```
ratio = exp(logπ_new - logπ_old)
```

So it **is** a ratio — just expressed in log-space first for stability, and then exponentiated to get back to ratio form.

### 🔁 Why use the ratio?

Because PPO wants to know:

> "How much more (or less) confident is the *new* policy about the action compared to the *old* one?"

* If `ratio > 1` → model now **likes the action more**
* If `ratio < 1` → model now **likes it less**

Multiply this by the reward to decide: should this change be encouraged?

### 🎯 PPO Loss Goal:

```python
loss = - ratio * reward  +  KL penalty  # or clipping
```

So the model is:

* ✅ Encouraged to increase π_new(action) if reward is high
* ❌ Penalized if it changes too much

### 🧠 TL;DR

| Term                  | Meaning                                     |
| --------------------- | ------------------------------------------- |
| `probs`               | π(action) — likelihood of chosen token      |
| `logprobs`            | log(π(action)) — used for stability         |
| `logπ_new - logπ_old` | Log of the ratio = log(π_new / π_old)       |
| `ratio`               | π_new / π_old — how much the policy changed |

✅ So yes — it's still a ratio.
You're just seeing it go through a `log → subtraction → exp` flow to keep the math clean and stable.







## more


# PPO Baseline (with Tiny GPT-Style Policy)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Tiny GPT-style Policy ===
class TinyPolicy(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)  # Predict next-token logits

    def forward(self, input_ids):
        x = self.embed(input_ids)          # [seq_len, hidden]
        last_hidden = x[-1]                # Use last token's embedding
        logits = self.lm_head(last_hidden) # [vocab_size]
        return logits  # logits over vocab for next token

# === Reward function: +1 if token == 1 ===
def fake_reward(token_ids):
    return torch.where(token_ids == 1, torch.tensor(1.0), torch.tensor(0.0))

# === PPO Loss Function ===
def ppo_loss(new_logprobs, old_logprobs, rewards, kl_coeff=0.1):
    ratio = torch.exp(new_logprobs - old_logprobs)
    surrogate = ratio * rewards
    kl = (old_logprobs - new_logprobs).mean()
    return -surrogate.mean() + kl_coeff * kl

# === Training Loop ===
vocab_size = 10
policy = TinyPolicy(vocab_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for step in range(300):
    prompt = torch.tensor([0])
    generated = [0]
    old_logprobs = []
    rewards = []

    for _ in range(8):
        logits = policy(prompt)
        probs = F.softmax(logits, dim=-1)
        # torch.multinomial adds controlled randomness — usually picks high-prob tokens like index 8  
        # but sometimes explores others — perfect for RLHF-style exploration vs argmax
        next_token = torch.multinomial(probs, num_samples=1)
        logprob = torch.log(probs[next_token])

        old_logprobs.append(logprob)
        rewards.append(fake_reward(next_token))
        generated.append(next_token.item())

        prompt = torch.cat([prompt, next_token], dim=0)

    generated_tensor = torch.tensor(generated[1:])
    rewards = torch.stack(rewards).squeeze()
    old_logprobs = torch.stack(old_logprobs).squeeze()

    new_logprobs = []
    prompt = torch.tensor([0])
    for token in generated_tensor:
        logits = policy(prompt)
        probs = F.softmax(logits, dim=-1)
        logprob = torch.log(probs[token])
        new_logprobs.append(logprob)
        prompt = torch.cat([prompt, token.unsqueeze(0)], dim=0)

    new_logprobs = torch.stack(new_logprobs)

    loss = ppo_loss(new_logprobs, old_logprobs, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        avg_reward = rewards.float().mean().item()
        print(f"Step {step} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.2f}")
```

# DPO Version (Direct Preference Optimization)

```python
# === DPO Loss ===
def dpo_loss(logprob_chosen, logprob_rejected):
    return -F.logsigmoid(logprob_chosen - logprob_rejected)

# === Paired Training Example ===
prompt = torch.tensor([0])
response_a = torch.tensor([2, 1, 3])
response_b = torch.tensor([2, 4, 3])
preferred = response_a  # Assume response_a is better
rejected = response_b

def compute_logprob(policy, prompt, response):
    input_seq = prompt.clone()
    logprobs = []
    for token in response:
        logits = policy(input_seq)
        probs = F.softmax(logits, dim=-1)
        logprobs.append(torch.log(probs[token]))
        input_seq = torch.cat([input_seq, token.unsqueeze(0)], dim=0)
    return torch.stack(logprobs).sum()

logprob_chosen = compute_logprob(policy, prompt, preferred)
logprob_rejected = compute_logprob(policy, prompt, rejected)
loss = dpo_loss(logprob_chosen, logprob_rejected)

loss.backward()
optimizer.step()
```

# SPO Version (Score-based Preference Optimization)

```python
# === SPO Loss ===
def spo_loss(logprob_a, logprob_b, score_a, score_b):
    score_diff = score_a - score_b
    logprob_diff = logprob_a - logprob_b
    return -score_diff * logprob_diff

# === Paired Training with Scores ===
score_a = torch.tensor(1.0)
score_b = torch.tensor(0.0)
logprob_a = compute_logprob(policy, prompt, response_a)
logprob_b = compute_logprob(policy, prompt, response_b)
loss = spo_loss(logprob_a, logprob_b, score_a, score_b)

loss.backward()
optimizer.step()
```

# GRPO Version (Generalized Reweighted PPO)

```python
# === GRPO Loss ===
def grpo_loss(new_logprob, old_logprob, preference, beta=1.0):
    ratio = torch.exp(new_logprob - old_logprob)
    weight = preference * ratio + (1 - preference)
    return -torch.log(weight)

# === Paired Training with Preference ===
# preference = 1 if response_a preferred, else 0
preference = torch.tensor(1.0)
old_logprob = logprob_chosen.detach()
new_logprob = logprob_chosen  # Recomputed with current policy
loss = grpo_loss(new_logprob, old_logprob, preference)

loss.backward()
optimizer.step()
```


# 🧠 RLHF Method Extensions: PPO, DPO, SPO, GRPO

This README expands your baby PPO implementation into modern preference-based fine-tuning methods used in RLHF research:

* **PPO**: Policy optimization with sampled rewards and logprob ratios
* **DPO**: Direct preference optimization using preferred vs rejected responses
* **SPO**: Score-based preference optimization with real-valued feedback
* **GRPO**: Generalized Reweighted PPO using preference-weighted logprob ratios

Each method keeps the same tiny GPT-style policy and PyTorch training loop style.

---

## ✅ PPO Baseline (with Tiny GPT-Style Policy)

This is the basic setup where we:

* Generate an 8-token sequence from a simple policy
* Compute token-level rewards
* Use the PPO loss to scale updates based on log-ratio of new vs old policy

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# === Tiny GPT-style Policy ===
class TinyPolicy(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)  # Predict next-token logits

    def forward(self, input_ids):
        x = self.embed(input_ids)          # [seq_len, hidden]
        last_hidden = x[-1]                # Use last token's embedding
        logits = self.lm_head(last_hidden) # [vocab_size]
        return logits  # logits over vocab for next token

# === Reward function: +1 if token == 1 ===
def fake_reward(token_ids):
    return torch.where(token_ids == 1, torch.tensor(1.0), torch.tensor(0.0))

# === PPO Loss Function ===
def ppo_loss(new_logprobs, old_logprobs, rewards, kl_coeff=0.1):
    ratio = torch.exp(new_logprobs - old_logprobs)
    surrogate = ratio * rewards
    kl = (old_logprobs - new_logprobs).mean()
    return -surrogate.mean() + kl_coeff * kl

# === Training Loop ===
vocab_size = 10
policy = TinyPolicy(vocab_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

for step in range(300):
    prompt = torch.tensor([0])
    generated = [0]
    old_logprobs = []
    rewards = []

    for _ in range(8):
        logits = policy(prompt)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        logprob = torch.log(probs[next_token])

        old_logprobs.append(logprob)
        rewards.append(fake_reward(next_token))
        generated.append(next_token.item())
        prompt = torch.cat([prompt, next_token], dim=0)

    generated_tensor = torch.tensor(generated[1:])
    rewards = torch.stack(rewards).squeeze()
    old_logprobs = torch.stack(old_logprobs).squeeze()

    new_logprobs = []
    prompt = torch.tensor([0])
    for token in generated_tensor:
        logits = policy(prompt)
        probs = F.softmax(logits, dim=-1)
        logprob = torch.log(probs[token])
        new_logprobs.append(logprob)
        prompt = torch.cat([prompt, token.unsqueeze(0)], dim=0)

    new_logprobs = torch.stack(new_logprobs)
    loss = ppo_loss(new_logprobs, old_logprobs, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        avg_reward = rewards.float().mean().item()
        print(f"Step {step} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.2f}")
```

> 🔁 PPO uses a reward-weighted ratio: `ratio = exp(logπ_new - logπ_old)`
> It encourages higher rewards but penalizes large policy shifts (via KL or clipping).

---

## ⚖️ DPO (Direct Preference Optimization)

DPO compares two responses and teaches the model to prefer the better one by increasing its logprob.

```python
# === DPO Loss ===
def dpo_loss(logprob_chosen, logprob_rejected):
    return -F.logsigmoid(logprob_chosen - logprob_rejected)

# === Paired Training Example ===
prompt = torch.tensor([0])
response_a = torch.tensor([2, 1, 3])
response_b = torch.tensor([2, 4, 3])
preferred = response_a
rejected = response_b

def compute_logprob(policy, prompt, response):
    input_seq = prompt.clone()
    logprobs = []
    for token in response:
        logits = policy(input_seq)
        probs = F.softmax(logits, dim=-1)
        logprobs.append(torch.log(probs[token]))
        input_seq = torch.cat([input_seq, token.unsqueeze(0)], dim=0)
    return torch.stack(logprobs).sum()

logprob_chosen = compute_logprob(policy, prompt, preferred)
logprob_rejected = compute_logprob(policy, prompt, rejected)
loss = dpo_loss(logprob_chosen, logprob_rejected)
loss.backward()
optimizer.step()
```

> ἟e἟9 DPO uses **pairwise comparisons** with no critic.
> It teaches the model to score preferred sequences higher.

---

## 📈 SPO (Score-based Preference Optimization)

SPO generalizes DPO by using real-valued scores instead of binary preferences.

```python
# === SPO Loss ===
def spo_loss(logprob_a, logprob_b, score_a, score_b):
    score_diff = score_a - score_b
    logprob_diff = logprob_a - logprob_b
    return -score_diff * logprob_diff

# === Paired Training with Scores ===
score_a = torch.tensor(1.0)
score_b = torch.tensor(0.0)
logprob_a = compute_logprob(policy, prompt, response_a)
logprob_b = compute_logprob(policy, prompt, response_b)
loss = spo_loss(logprob_a, logprob_b, score_a, score_b)
loss.backward()
optimizer.step()
```

> 🧹 SPO uses **continuous reward scores** to shape policy gradients directly.

---

## 🔄 GRPO (Generalized Reweighted PPO)

GRPO keeps the PPO-style ratio but replaces rewards with preference indicators.

```python
# === GRPO Loss ===
def grpo_loss(new_logprob, old_logprob, preference, beta=1.0):
    ratio = torch.exp(new_logprob - old_logprob)
    weight = preference * ratio + (1 - preference)
    return -torch.log(weight)

# === Paired Training with Preference ===
preference = torch.tensor(1.0)
old_logprob = logprob_chosen.detach()
new_logprob = logprob_chosen
loss = grpo_loss(new_logprob, old_logprob, preference)
loss.backward()
optimizer.step()
```

> ⚖️ GRPO bridges PPO and DPO: it reweights PPO by preference without needing a critic.

---

These examples form a clean playground for building, testing, and comparing RLHF-style fine-tuning strategies in PyTorch. You can plug in real reward models, critics, or human preference data later.

Let me know if you'd like a batched version, visualization, or `torch.compile`-ready code.

