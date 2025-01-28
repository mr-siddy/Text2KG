# src/inference.py

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def generate_triplets(model_path: str, prompt: str, max_length: int = 256):
    """
    Loads the distilled student model from model_path and uses it to generate
    triplets for the provided prompt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="float16"
    )
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            do_sample=False  # or True if you want sampling
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def run_inference_example():
    model_path = "output-distilled"  # wherever your distilled model is saved
    prompt = (
        "You are a relation extraction assistant. "
        "Extract relevant relations in the form of triplets for the given text: "
        "'Nutrition impacts health by improving diets and energy levels. Poor diet can lead to health issue'. "
        "Provide the result as {source} | {relation} | {target}."
    )

    triplets = generate_triplets(model_path, prompt)
    print("Generated triplets:\n", triplets)

if __name__ == "__main__":
    run_inference_example()
