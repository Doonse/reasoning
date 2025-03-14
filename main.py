from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "AlibabaResearch/qwen-2b"  # hypothetical

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, trust_remote_code=True)

