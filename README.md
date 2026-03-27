# E-commerce Distributed SQL Fine-tuned Model

Fine-tuned Llama 3.2 3B on e-commerce distributed SQL queries using QLoRA.

## Model on HuggingFace
https://huggingface.co/haricharanhl22/ecommerce-distributed-sql

## What it does
Converts natural language to distributed SQL queries for e-commerce scenarios.

## Example
Input: "Find all customers who spent more than 1000 euros in Germany"
Output: SELECT * FROM customers WHERE country = 'Germany' AND amount > 1000;

## Tech Stack
- HuggingFace Transformers
- PEFT / QLoRA
- TRL
- Google Colab T4 GPU (free)