from transformers import AutoTokenizer, AutoModelForCausalLM

def get_tokenizer(checkpoint_name: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    return tokenizer

def get_model(checkpoint_name: str):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_name)

    return model