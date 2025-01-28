# src/modeling.py

from transformers import LlamaForCausalLM

def get_teacher_model(teacher_model_name_or_path: str):
    """
    Loads the teacher model (70B).
    """
    teacher_model = LlamaForCausalLM.from_pretrained(
        teacher_model_name_or_path,
        device_map="auto",  # or custom device map for multi-GPU
        torch_dtype="float16"
    )
    return teacher_model

def get_student_model(student_model_name_or_path: str):
    """
    Loads the student model (8B).
    """
    student_model = LlamaForCausalLM.from_pretrained(
        student_model_name_or_path,
        device_map="auto",
        torch_dtype="float16"
    )
    return student_model
