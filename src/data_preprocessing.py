# src/data_preprocessing.py

import json
import os
from typing import List, Dict

def load_enriched_data(json_path: str) -> List[Dict]:
    """
    Loads the enriched data JSON which contains chunk_text, NER entities, and relation_triplets.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_instruction_qa(json_path: str) -> List[Dict]:
    """
    Loads the instruction_qa data JSON which contains instruction, question, and answer.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_text_to_kg_samples(
    enriched_data: List[Dict],
    instruction_qa_data: List[Dict]
) -> List[Dict]:
    """
    Merges/Prepares a list of data samples for the text-to-KG distillation tasks.
    Each sample might look like:
      {
        "input_text": [concatenated text or instruction + question],
        "target_text": [the desired triplet output or label text],
      }
    """
    samples = []

    # Approach 1: Use instruction_qa directly
    for item in instruction_qa_data:
        # Combine instruction + question as "prompt", 
        # answer is the "label" to predict
        prompt = f"{item['Instruction']}\n\n{item['Question']}"
        label = item['Answer']

        samples.append({
            "input_text": prompt,
            "target_text": label
        })

    # Approach 2: Optionally also incorporate the raw enriched_data 
    # if you want to create more training examples from chunk_text and triplets 
    # (this is up to you).
    # for doc in enriched_data:
    #     text = doc["chunk_text"]
    #     # You might want to generate custom instructions/labels
    #     # ...
    #     # samples.append({...})

    return samples
