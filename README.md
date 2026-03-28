# Fine-tuned LLM for Distributed SQL

> Live model: [huggingface.co/haricharanhl22/ecommerce-distributed-sql](https://huggingface.co/haricharanhl22/ecommerce-distributed-sql)

Fine-tuned Llama 3.2 3B on e-commerce distributed SQL queries using QLoRA. The model converts natural language questions into SQL queries optimized for distributed database scenarios.

## Example

**Input:**
```
Find all customers who spent more than 1000 euros in Germany
```

**Output:**
```sql
SELECT * FROM customers 
WHERE country = 'Germany' AND amount > 1000;
```

## What is QLoRA?

Instead of retraining all 3 billion parameters (requires 80GB GPU, weeks of time), QLoRA uses two tricks:
- **Quantization** — compresses model from 32-bit to 4-bit (8x less memory)
- **LoRA** — adds tiny trainable adapter layers, freezes the rest

Result: trained on a **free Google Colab T4 GPU in 20 minutes**, training only **0.14% of parameters**.

## Tech Stack

- **Base model** — Llama 3.2 3B (Meta)
- **HuggingFace Transformers** — model loading and inference
- **PEFT** — LoRA adapter implementation
- **TRL + SFTTrainer** — supervised fine-tuning
- **bitsandbytes** — 4-bit quantization
- **Google Colab** — free T4 GPU training
- **HuggingFace Hub** — model hosting

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Llama 3.2 3B |
| Training method | QLoRA (4-bit) |
| LoRA rank | 16 |
| Trainable params | 0.14% |
| Dataset size | 25 examples |
| Epochs | 3 |
| GPU | Google Colab T4 (free) |
| Training time | ~20 minutes |

## Dataset

25 natural language → SQL pairs covering e-commerce scenarios:
- Orders across regions
- Inventory across warehouses  
- Customer analytics
- Revenue by segment

## Use the Model
```python
from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

pipe = pipeline("text-generation", 
    model="haricharanhl22/ecommerce-distributed-sql")

query = """### Instruction:
Convert to distributed SQL

### Input:
Find top 5 customers by revenue

### Response:"""

result = pipe(query, max_new_tokens=100, do_sample=False)
print(result[0]["generated_text"])
```

## Author

**Hari Charan Hosakote Lokesh**
- GitHub: [@haricharanhl22](https://github.com/haricharanhl22)
- LinkedIn: [haricharanhl22](https://linkedin.com/in/haricharanhl22)
- HuggingFace: [haricharanhl22](https://huggingface.co/haricharanhl22)
