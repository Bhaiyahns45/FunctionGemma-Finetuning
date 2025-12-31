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
    - Supervised Fine-Tuning (SFT) using `SFTTrainer`.
    - Post-training evaluation and comparison.
    - Automated visualization of training progress and agent-specific accuracy.


## ⚙️ Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Epochs** | 8 |
| **Batch Size** | 4 |
| **Learning Rate** | 5e-5 |
| **Optimizer** | `adamw_torch_fused` |
| **Mixed Precision** | `fp16` (on T4) |
| **Scheduler** | `constant` |

## 📊 Evaluation & Results
<img width="1489" height="1200" alt="newplot" src="https://github.com/user-attachments/assets/b667e27a-aa68-44b4-a362-ce56a08e9e6e" />

<img width="1374" height="551" alt="image" src="https://github.com/user-attachments/assets/5877e2f2-794c-4f8b-8f8d-f18034ea0eb5" />


## 🤝 Contribution

Contributions are welcome! If you have ideas for adding more specialized agents or improving the routing logic, feel free to open a PR.
