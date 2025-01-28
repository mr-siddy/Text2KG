# src/run_distillation.py

import os
from transformers import LlamaTokenizer

from data_preprocessing import (
    load_enriched_data,
    load_instruction_qa,
    prepare_text_to_kg_samples
)
from dataset import TextToKGDataset
from modeling import get_teacher_model, get_student_model
from training import train_distillation

def main():
    # 1. Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    enriched_data_path = os.path.join(data_dir, "enriched_data.json")
    instruction_qa_path = os.path.join(data_dir, "instruction_qa.json")

    enriched_data = load_enriched_data(enriched_data_path)
    instruction_qa = load_instruction_qa(instruction_qa_path)

    samples = prepare_text_to_kg_samples(enriched_data, instruction_qa)

    # 2. Load tokenizer (assuming you have a LlamaTokenizer for your 70B & 8B)
    #    In practice, you might be using the same tokenizer or a custom one.
    teacher_model_name = "path_or_name_for_llama_70B"   # e.g. "meta-llama/Llama-2-70b-hf"
    student_model_name = "path_or_name_for_llama_8B"    # e.g. "meta-llama/Llama-2-7b-hf" or your custom 8B

    tokenizer = LlamaTokenizer.from_pretrained(teacher_model_name)

    # 3. Build Dataset
    train_dataset = TextToKGDataset(samples, tokenizer, max_length=512)

    # 4. Load teacher/student models
    teacher_model = get_teacher_model(teacher_model_name)
    student_model = get_student_model(student_model_name)

    # 5. Distill
    train_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        output_dir="output-distilled",
        epochs=1,
        batch_size=2,
        lr=1e-5,
        warmup_ratio=0.1,
        alpha_ce=0.5,
        alpha_mle=0.5,
        temperature=2.0
    )

if __name__ == "__main__":
    main()
