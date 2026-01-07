# FunctionGemma Multi-Agent Routing Fine-tuning

This repository contains a specialized implementation for fine-tuning(STF)  **FunctionGemma** (270M parameters) to act as an intelligent routing layer in a multi-agent ecosystem.

## 🚀 Projects Overview

The system is designed to route customer support queries to one of three specialized agents based on the content and intent of the user's message. It leverages the function-calling capabilities of FunctionGemma to decide which agent should handle the request and with what parameters.

### Specialized Agents

1.  **`technical_support_agent`**: Handles technical issues, bugs, troubleshooting, and API integrations.
    - *Parameters*: `issue_type`, `priority`.
2.  **`billing_agent`**: Manages payments, invoices, subscriptions, refunds, and pricing inquiries.
    - *Parameters*: `request_type`, `urgency`.
3.  **`product_info_agent`**: Provides product details, feature comparisons, limits, and compliance information.
    - *Parameters*: `query_type`, `category`.

## 🛠️ Technical Stack

- **Model**: `google/functiongemma-270m-it`
- **Libraries**: `torch`, `transformers`, `datasets`, `trl`, `accelerate`, `evaluate`.
- **Optimization**: Configured for **Free T4 GPU (16GB VRAM)** environments like Google Colab.
- **Workflow**:
    - Synthetic dataset generation (conversational format).
    - Pre-training evaluation (baseline accuracy).
    - Supervised Fine-Tuning (SFT) using `SFTTrainer` with improved hyperparameters.
    - Post-training evaluation and comparison.
    - Automated visualization of training progress and agent-specific accuracy.
    - Dataset and model upload to Hugging Face Hub with comprehensive documentation.


## ⚙️ Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 15 | Converges around epoch 8-10 (slower convergence) |
| **Batch Size** | 4 | Per device (effective: 12 with gradient accumulation) |
| **Learning Rate** | 1e-5 | Lower rate for slower convergence (8-10 epochs) |
| **Optimizer** | `adamw_torch_fused` | Fused AdamW for efficiency |
| **Weight Decay** | 0.02 | Higher L2 regularization to slow convergence |
| **Warmup Ratio** | 0.05 | 5% warmup for slower start |
| **Scheduler** | `cosine` | Cosine annealing for gradual convergence |
| **Mixed Precision** | `fp16` (on T4) | Automatic based on model dtype |
| **Gradient Accumulation** | 3 | Effective batch size = 12 (larger for stability) |
| **Train/Test Split** | 92/23 | ~92 train, ~23 test (expanded dataset) |
| **Dataset Size** | 115 samples | Expanded with diverse examples for balanced distribution |

## 📊 Evaluation & Results

### Training Results

The model achieved significant improvements after fine-tuning:

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| **Overall Accuracy** | 4.3% (1/23) | **82.6% (19/23)** | **+78.3%** |
| **Correct Predictions** | 1 | **19** | **+18** |

### Agent-Specific Performance

| Agent | Before | After | Improvement |
|-------|--------|-------|-------------|
| 🔧 **Technical Support** | 0% | **100%** | **+100%** |
| 💰 **Billing** | 0% | **80%** | **+80%** |
| 📊 **Product Info** | 12% | **75%** | **+62%** |

### Training Configuration for Slower Convergence

The configuration is optimized for **slower convergence** (8-10 epochs instead of 4):

- ✅ **Lower learning rate** (1e-5) to slow down learning
- ✅ **Higher weight decay** (0.02) for more regularization
- ✅ **Reduced warmup** (5%) for slower start
- ✅ **Larger effective batch** (12) with gradient accumulation
- ✅ **Expanded dataset** (~115 samples) for more diversity
- ✅ **15 epochs** with convergence expected around epoch 8-10

### Training Visualizations

<img width="1198" height="361" alt="image" src="https://github.com/user-attachments/assets/214f0338-a864-43a8-aba6-7d6c6b5b5e42" />
<img width="1083" height="325" alt="image" src="https://github.com/user-attachments/assets/1b8e5699-36cb-403d-a17e-60aa474f049f" />
<img width="1116" height="326" alt="image" src="https://github.com/user-attachments/assets/d9c22c3c-3f98-452b-a51f-4883114094b5" />






<img width="1002" height="559" alt="image" src="https://github.com/user-attachments/assets/cb11d935-1313-4f3b-88f7-d6502c6dd566" />




## 📈 Performance Analysis

### Current Results
- **Overall Accuracy**: 82.6% (19/23 correct)
- **Best Performing Agent**: Technical Support (100% accuracy)
- **Areas for Improvement**: Product Info agent (75% - lowest accuracy)

### Performance Tips

**Current Configuration:**
- **Dataset**: 115 samples with balanced distribution across all agents
- **Convergence**: Model converges around epoch 8-10 (slower than before)
- **Training**: More gradual learning curve with better generalization
- **Diversity**: Includes varied query types, edge cases, and realistic scenarios

**For even better results (target: 85-95% accuracy):**
- **Monitor convergence**: Should see gradual loss decrease over 8-10 epochs
- **If converging too fast**: Further reduce learning rate to 5e-6 or increase weight decay to 0.03
- **If converging too slow**: Increase learning rate to 1.5e-5 or reduce weight decay to 0.015
- **Expand dataset further**: Add more product-info-related queries (currently weakest at 75%)
- **Data augmentation**: Paraphrase existing queries to increase diversity
- **Monitor**: Track per-agent performance to identify weak areas

## 📚 Resources

- **Model on Hugging Face**: [bhaiyahnsingh45/functiongemma-multiagent-router](https://huggingface.co/bhaiyahnsingh45/functiongemma-multiagent-router)
- **Dataset on Hugging Face**: [bhaiyahnsingh45/multiagent-router-finetuning](https://huggingface.co/datasets/bhaiyahnsingh45/multiagent-router-finetuning)
- **Base Model**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)

## 🤝 Contribution

Contributions are welcome! If you have ideas for:
- Adding more specialized agents
- Improving the routing logic
- Expanding the dataset
- Optimizing training parameters

Feel free to open a PR or issue!
