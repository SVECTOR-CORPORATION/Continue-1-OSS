<p align="center">
  <img alt="Continue-1-OSS" src="https://github.com/SVECTOR-CORPORATION/Continue-1-OSS/blob/main/Continue-1-OSS-image-banner.jpg?raw=true" width="800">
</p>

# Continue-1-OSS

### Advanced Text Generation Model

<div align="left" style="line-height: 1;">
  <a href="https://spec-chat.tech" target="_blank" style="margin: 2px;">
    <img alt="SVECTOR" src="https://img.shields.io/badge/💬%20Spec%20Chat-Spec%20Chat-blue?style=plastic" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <a href="https://huggingface.co/SVECTOR-CORPORATION" target="_blank" style="margin: 2px;">
    <img alt="SVECTOR" src="https://img.shields.io/badge/🤗%20Hugging%20Face-SVECTOR-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <a href="https://huggingface.co/SVECTOR-CORPORATION/Continue-1-OSS/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue?color=1e88e5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <a href="https://github.com/SVECTOR-CORPORATION/Continue-1-OSS" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Continue--1--OSS-181717?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction

We are thrilled to introduce **Continue-1-OSS**, an advanced text generation model developed by SVECTOR, built on the Continue-1 architecture optimized for high-quality text generation, instruction following, and long-context understanding.

**Continue-1-OSS** is engineered to provide:

- **Superior Instruction Following:** Accurately follows complex, multi-step instructions
- **Long Context:** Robust handling of up to 128K+ tokens
- **Natural Conversations:** Human-like dialogue with strong reasoning capabilities
- **Tool Integration:** Built-in support for function calling and external tool use
- **Open Source:** Fully accessible under Apache 2.0 license for research and commercial use

This model combines the power of transformer architecture with advanced training techniques to deliver exceptional performance across a wide range of natural language tasks.

### Model Specifications

- **Base Architecture:** Continue1ForCausalLM (transformer decoder)
- **Model Type:** continue_oss
- **Parameters:** 3 Billion
- **Context Length:** 131,072 tokens
- **Vocabulary Size:** 128,256 tokens
- **Hidden Size:** 3072
- **Number of Layers:** 28
- **Attention Heads:** 24
- **License:** Apache 2.0


## Requirements

To use Continue-1-OSS, install the required dependencies:

```bash
pip install transformers torch
pip install vllm  # For fast inference (optional but recommended)
```

## Quickstart

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "SVECTOR-CORPORATION/Continue-1-OSS"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare conversation
messages = [
    {"role": "user", "content": "What is machine learning?"}
]

# Apply chat template and generate
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using vLLM (Recommended for Production)

For high-performance inference with faster generation:

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="SVECTOR-CORPORATION/Continue-1-OSS",
    trust_remote_code=True,
    max_model_len=8192
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Generate
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

outputs = llm.chat(messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

**Default System Prompt:** "You are Continue-1-OSS, an advanced AI assistant developed by SVECTOR. You are designed to be helpful, harmless, and honest."

## Advanced Features

### Multi-Turn Conversations

```python
messages = [
    {"role": "system", "content": "You are Continue-1-OSS, a helpful AI assistant."},
    {"role": "user", "content": "What is quantum computing?"},
    {"role": "assistant", "content": "Quantum computing is a type of computing that uses quantum mechanics principles..."},
    {"role": "user", "content": "Can you explain that more simply?"}
]
```

### Tool Calling Support

Continue-1-OSS supports function calling for tool integration:

```python
messages = [
    {"role": "user", "content": "What's the weather in San Francisco?"}
]

# Model can generate JSON function calls
# Example output: {"name": "get_weather", "parameters": {"location": "Ahmedabad"}}
```


## Use Cases

Continue-1-OSS excels at:

- **Conversational AI:** Build chatbots and virtual assistants with natural dialogue
- **Content Generation:** Generate articles, stories, and creative content
- **Code Assistance:** Help with coding tasks, debugging, and code explanations
- **Question Answering:** Answer questions based on context with high accuracy
- **Summarization:** Condense long documents into concise summaries
- **Data Extraction:** Extract structured data from unstructured text
- **Tool Integration:** Call functions and use external tools intelligently
- **Education:** Create educational content and tutoring assistance
- **Customer Service:** Automated support with natural language understanding

## Performance

- **Quality:** State-of-the-art instruction following and text generation
- **Speed:** Fast inference with vLLM optimization
- **Memory:** ~7GB GPU RAM (BF16), ~14GB (FP32)
- **Context:** Handles up to 128K tokens effectively
- **Efficiency:** Competitive with much larger models on many tasks

## Model Architecture

Continue-1-OSS uses a custom architecture based on the transformer decoder:

- **Architecture Class:** `Continue1ForCausalLM`
- **Config Class:** `Continue1Config`
- **Hidden Size:** 3072
- **Num Layers:** 28
- **Num Attention Heads:** 24
- **Intermediate Size:** 8192
- **Vocab Size:** 128,256
- **Max Position Embeddings:** 131,072

The model uses RoPE (Rotary Position Embeddings) for positional encoding and supports extended context through position interpolation.

## Training

Continue-1-OSS was developed using:
- High-quality instruction datasets covering diverse tasks
- Conversational and reasoning data for improved dialogue
- Code and technical content for developer assistance
- Multi-turn dialogue for contextual understanding

Training utilized:
- Advanced optimization techniques
- Careful hyperparameter tuning
- Quality filtering and data curation
- Evaluation on diverse benchmarks


## Limitations

As with any language model, Continue-1-OSS has certain limitations:

- **Knowledge Cutoff:** Training data is limited to information available up to December 2023
- **Factual Accuracy:** May occasionally generate incorrect or outdated information
- **Specialized Domains:** Performance may vary on highly specialized technical knowledge
- **Long Context:** Very long contexts (>64K tokens) may impact generation quality
- **Languages:** Primarily optimized for English; other languages have limited support
- **Reasoning:** Complex multi-step reasoning may require careful prompting
- **Compute:** Requires GPU for optimal performance (CPU is significantly slower)

## Ethical Considerations

SVECTOR is committed to responsible AI development. Users should:

- **Transparency:** Disclose when content is AI-generated
- **Verification:** Always fact-check important information generated by the model
- **Bias Awareness:** Be aware the model may reflect biases present in training data
- **Privacy:** Do not input personal or sensitive information without proper safeguards
- **Safety:** Implement content filtering and guardrails for production applications
- **Responsible Use:** Do not use for illegal purposes, misinformation, or harmful content
- **Attribution:** Credit the model when used in public projects or research

## Performance Tips

1. **Temperature Settings:**
   - 0.0-0.3 for factual/deterministic tasks
   - 0.7-0.9 for creative tasks
   
2. **Context Management:**
   - Model supports 128K tokens but consider truncating for faster inference
   - Use sliding window for very long documents

3. **Batch Processing:**
   - Use vLLM for efficient batched inference in production
   - Group similar-length prompts together

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "SVECTOR-CORPORATION/Continue-1-OSS",
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto"
)
```


## License

This model is released under the **Apache License 2.0**. You are free to use, modify, and distribute this model for both commercial and non-commercial purposes. See the [LICENSE](https://huggingface.co/SVECTOR-CORPORATION/Continue-1-OSS/blob/main/LICENSE) file for complete details.

---

<p align="center">
    <i>Developed by <a href="https://www.svector.co.in">SVECTOR</a></i>
</p>
