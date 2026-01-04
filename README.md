# FunctionGemma Multi-Agent Routing Fine-tuning

This repository contains a specialized implementation for fine-tuning **FunctionGemma** (270M parameters) to act as an intelligent routing layer in a multi-agent ecosystem.

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
| **Epochs** | 15 | With early stopping based on validation loss |
| **Batch Size** | 4 | Per device (effective: 8 with gradient accumulation) |
| **Learning Rate** | 2e-5 | Lower rate for stable training and better generalization |
| **Optimizer** | `adamw_torch_fused` | Fused AdamW for efficiency |
| **Weight Decay** | 0.01 | L2 regularization to prevent overfitting |
| **Warmup Ratio** | 0.1 | 10% of training steps for learning rate warmup |
| **Scheduler** | `cosine` | Cosine annealing for better convergence |
| **Mixed Precision** | `fp16` (on T4) | Automatic based on model dtype |
| **Gradient Accumulation** | 2 | Effective batch size = 8 |
| **Train/Test Split** | 80/20 | More training data for small dataset |

## 📊 Evaluation & Results

### Training Results

The model achieved significant improvements after fine-tuning:

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| **Overall Accuracy** | 0.0% (0/13) | **76.9% (10/13)** | **+76.9%** |
| **Correct Predictions** | 0 | **10** | **+10** |

### Agent-Specific Performance

| Agent | Before | After | Improvement |
|-------|--------|-------|-------------|
| 🔧 **Technical Support** | 0% | **100%** | **+100%** |
| 💰 **Billing** | 0% | **60%** | **+60%** |
| 📊 **Product Info** | 0% | **75%** | **+75%** |

### Training Improvements

The optimized training configuration achieved these results:

- ✅ **Lower learning rate** (2e-5) for stable training
- ✅ **Weight decay** (0.01) to prevent overfitting
- ✅ **Cosine scheduler with warmup** for better convergence
- ✅ **Gradient accumulation** (effective batch size = 8)
- ✅ **More epochs** (15) with early stopping
- ✅ **Better train/test split** (80/20 instead of 70/30)

### Training Visualizations

<img width="1006" height="393" alt="image" src="https://github.com/user-attachments/assets/1ae3af69-acb2-4881-85ac-07d886e27b15" />
<img width="944" height="341" alt="image" src="https://github.com/user-attachments/assets/f3e371ee-00bc-4ec0-83a8-de3267e6b2b7" />
<img width="955" height="333" alt="image" src="https://github.com/user-attachments/assets/2c45b9c4-3be4-4c45-8626-cdd3dee9e64d" />




<img width="987" height="550" alt="image" src="https://github.com/user-attachments/assets/b608018f-f308-4ca0-9cfd-b88634304a39" />



## 📈 Performance Analysis

### Current Results
- **Overall Accuracy**: 76.9% (10/13 correct)
- **Best Performing Agent**: Technical Support (100% accuracy)
- **Areas for Improvement**: Billing agent (60% - needs more training data)

### Performance Tips

For even better results (target: 85-95% accuracy):
- **Expand dataset** to 150-200+ samples (currently 65)
  - Add more billing-related queries to improve billing agent accuracy
  - Include more edge cases and query variations
- **Data augmentation**: Paraphrase existing queries to increase diversity
- **Fine-tuning**: Continue training if validation loss keeps decreasing
- **Monitor**: Track per-agent performance to identify weak areas

## 📚 Resources

- **Model on Hugging Face**: [bhaiyasingh45/functiongemma-multiagent-router](https://huggingface.co/bhaiyasingh45/functiongemma-multiagent-router)
- **Dataset on Hugging Face**: [bhaiyasingh45/multiagent-router-finetuning](https://huggingface.co/datasets/bhaiyasingh45/multiagent-router-finetuning)
- **Base Model**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)

## 🤝 Contribution

Contributions are welcome! If you have ideas for:
- Adding more specialized agents
- Improving the routing logic
- Expanding the dataset
- Optimizing training parameters

Feel free to open a PR or issue!
