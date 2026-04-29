# VideoLLM-Online Model Training and Inference Flow

This note summarizes how the model is built, trained, and used for streaming inference in this repo.

## 1. Overall Model Structure

The full model is:

```text
vision encoder
    -> connector MLP
    -> LLM embedding space
    -> LlamaForCausalLM
```

The important idea is that video frames are not passed into the LLM as ordinary token ids. The text prompt contains special `<v>` placeholder tokens, and those placeholder embedding positions are replaced with visual embeddings.

## 2. Model Construction

Training starts from:

```text
train.py
```

The model and tokenizer are created here:

```python
model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
```

`build_model_and_tokenizer` comes from:

```text
models/__init__.py
```

It aliases:

```python
from .live_llama import build_live_llama as build_model_and_tokenizer
```

Then `build_live_llama()` in:

```text
models/live_llama/modeling_live_llama.py
```

calls the shared builder:

```python
def build_live_llama(**kwargs):
    return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)
```

The actual LLM is loaded in:

```text
models/modeling_live.py
```

with:

```python
model = model_class.from_pretrained(
    llm_pretrained,
    config=config_class.from_pretrained(llm_pretrained, **kwargs),
    ...
)
```

The default model names are defined in:

```text
models/arguments_live.py
```

```python
llm_pretrained = 'meta-llama/Meta-Llama-3-8B-Instruct'
vision_pretrained = 'google/siglip-large-patch16-384'
```

So by default:

```text
LLM:    Meta-Llama-3-8B-Instruct
Vision: google/siglip-large-patch16-384
```

## 3. Vision Model

The vision encoder is loaded in:

```text
models/vision_live.py
```

```python
model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
```

For the default config, this is the `vision_model` part of SigLIP.

During training, the project usually uses pre-extracted visual features for efficiency. During demo inference, the vision model is placed inside the model object by passing:

```python
set_vision_inside=True
```

in:

```text
demo/inference.py
```

That calls:

```python
self.vision_encoder, self.vision_encode = build_live_vision(self.config)
```

from:

```text
models/modeling_live.py
```

## 4. Connector MLP

The connector MLP is defined in:

```text
models/live_llama/modeling_live_llama.py
```

```python
self.connector = torch.nn.Sequential(
    torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
    GELUActivation(config.hidden_size),
    torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
)
```

Its job is to project visual features into the Llama hidden dimension.

The flow is:

```text
raw frames or pre-extracted visual features
    -> visual_embed()
    -> connector MLP
    -> LLM-sized visual embeddings
```

## 5. Replacing `<v>` Token Embeddings

The replacement happens in:

```text
models/modeling_live.py
```

inside `joint_embed()`:

```python
inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
v_mask = input_ids == self.config.v_placeholder_id
if v_mask.any():
    inputs_embeds[v_mask] = self.visual_embed(frames)
```

So the LLM receives a normal embedding sequence, but the positions corresponding to `<v>` are visual embeddings instead of text-token embeddings.

## 6. Training Data Flow

Training dataset construction happens in:

```text
train.py
```

```python
train_dataset = build_concat_train_dataset(...)
data_collator = get_data_collator(...)
```

Dataset samples are prepared through:

```text
data/stream.py
```

Each sample returns roughly:

```text
text, frames, learn_ranges
```

For training, `frames` are often pre-extracted visual features loaded from `.pt` files:

```python
frames = torch.cat([torch.load(path, weights_only=True)[ranger] for path, ranger in load_ranges.items()])
```

The conversation is converted into text with the tokenizer chat template:

```python
text = self.tokenizer.apply_chat_template(...)
```

The supervised regions are computed with:

```python
learn_ranges = self.tokenizer.get_learn_ranges(conversation)
```

The `learn_ranges` logic lives in:

```text
models/tokenization_live.py
```

It marks two main types of learnable positions:

```text
1. assistant response text
2. stream / frame timing tokens around visual placeholders
```

## 7. Label Construction

Labels are created in:

```text
data/data_collator.py
```

The collator first fills everything with `-100`, meaning "ignore this position in the loss":

```python
batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
```

Then only the learnable ranges are filled:

```python
labels[start-1:stop-1] = input_ids[start:stop]
```

This performs the causal LM shift:

```text
current position predicts next token
```

The model therefore receives:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "frames": ...
}
```

## 8. Training Forward Pass

The forward pass is defined in:

```text
models/live_llama/modeling_live_llama.py
```

First it builds multimodal embeddings:

```python
if inputs_embeds is None:
    inputs_embeds = self.joint_embed(input_ids, frames)
```

Then it calls the original Llama forward:

```python
outputs = super().forward(
    inputs_embeds=inputs_embeds,
    ...
)
```

## 9. Loss

The loss is also defined in:

```text
models/live_llama/modeling_live_llama.py
```

```python
v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
weight = v_mask * self.config.stream_loss_weight + ~v_mask
loss = nn.functional.cross_entropy(
    logits.flatten(0, 1),
    labels.flatten(),
    reduction='none'
) * weight
loss = loss.sum() / (labels >= 0).sum()
```

Meaning:

```text
assistant text token loss: weight = 1
stream / <v> position loss: weight = stream_loss_weight
ignored positions: labels = -100
```

The overall loss is a weighted token-level cross entropy:

```text
total loss = text LM loss + stream_loss_weight * stream timing loss
```

In implementation, these are not computed as two separate loss tensors. The code computes one token-level cross entropy and weights the `<v>` positions.

During training, HuggingFace `Trainer` reads `outputs.loss` and backpropagates it.

LoRA is configured in:

```text
models/modeling_live.py
```

The default trainable parts are mainly:

```text
1. connector MLP
2. LoRA modules injected into Llama projection layers / lm_head
```

## 10. Inference Setup

Streaming inference is implemented in:

```text
demo/inference.py
```

The `LiveInfer` class loads:

```text
1. Llama base model
2. tokenizer and `<v>` token
3. PEFT / LoRA checkpoint
4. vision encoder
5. connector MLP
```

The model is created with:

```python
self.model, self.tokenizer = build_model_and_tokenizer(
    is_training=False,
    set_vision_inside=True,
    **asdict(args)
)
```

The checkpoint is provided by:

```bash
--resume_from_checkpoint ...
```

For example, the README uses:

```bash
--resume_from_checkpoint chenjoya/videollm-online-8b-v1plus
```

## 11. Streaming Video Inference

Video is loaded in:

```text
demo/inference.py
```

```python
self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0].to('cuda')
```

As video time moves forward, new frames are encoded:

```python
frames_embeds = self.model.visual_embed(self.video_tensor[ranger])
```

This runs:

```text
video frame
    -> vision encoder
    -> connector MLP
    -> frame embeddings
```

The frame embeddings are pushed into a queue:

```python
self.frame_embeds_queue.extend(...)
```

## 12. Streaming Decision Logic

The core streaming logic is in:

```text
demo/inference.py
```

inside `_call_for_streaming()`.

For each new frame, it concatenates:

```text
previous text token embeddings + current frame embeddings
```

with:

```python
inputs_embeds = torch.cat([
    self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
    frame_embeds.view(1, -1, self.hidden_size),
], dim=1)
```

Then it calls the model with KV cache:

```python
outputs = self.model(
    inputs_embeds=inputs_embeds,
    use_cache=True,
    past_key_values=self.past_key_values
)
```

Because `past_key_values` is reused, the model performs incremental streaming inference instead of recomputing the full history every time.

The model predicts the next token. If the next token is `frame_token_interval_id`, it means:

```text
keep waiting for the next frame
```

If the next token is not `frame_token_interval_id`, it means:

```text
start generating an assistant response now
```

## 13. Response Generation

When the model decides to answer, generation happens in:

```text
models/modeling_live.py
```

via:

```python
fast_greedy_generate(...)
```

This repeatedly:

```text
1. calls the model with the current token embedding and KV cache
2. selects argmax token
3. appends it to output
4. stops when eos is produced
```

## 14. One-Line Summary

Training:

```text
conversation + pre-extracted visual features
    -> chat template with `<v>` placeholders
    -> labels from learn_ranges
    -> replace `<v>` embeddings with visual embeddings
    -> Llama forward
    -> weighted CE loss for text and stream timing
    -> update LoRA + connector
```

Inference:

```text
video frames
    -> vision encoder
    -> connector MLP
    -> frame embeddings
    -> append to LLM context with KV cache
    -> decide whether to wait or answer
    -> greedy generate response
```

