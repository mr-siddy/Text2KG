import json
from datasets import Dataset, DatasetDict
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# --- Data Loading and Transformation ---

def load_dataset(enriched_data_path, instruction_qa_path):
    # Load both JSON files
    with open(enriched_data_path, 'r') as f:
        enriched_data = json.load(f)
    with open(instruction_qa_path, 'r') as f:
        instruction_data = json.load(f)
    
    transformed_data = []
    
    # Process enriched data: add chain-of-thought with named entity information.
    for entry in enriched_data:
        text = entry['chunk_text']
        # Create a string that summarizes all NER information.
        ner_info = ", ".join(
            [f"{ne['entity']} ({ne['label']}, score: {ne['score']:.2f})" 
             for ne in entry['ner_entities']]
        )
        
        # For each relation triplet, create a CoT style prompt and answer.
        for triplet in entry['relation_triplets']:
            instruction = "Extract relations from the provided text with chain-of-thought reasoning."
            question = (
                f"Given the text: \"{text}\" \n"
                f"and the following named entity information: {ner_info}, \n"
                "first analyze the entities and then extract the relation triplets in the form {source} | {relation} | {target}."
            )
            # The answer is crafted to illustrate a chain-of-thought style reasoning:
            answer = (
                "Step 1: Identify Named Entities from the text.\n"
                f"  -> Recognized Entities: {ner_info}\n"
                "Step 2: Analyze relations between these entities based on context.\n"
                f"  -> Extracted Relation: {triplet['source']} | {triplet['relation']} | {triplet['target']}\n"
                "Final Answer: "
                f"{triplet['source']} | {triplet['relation']} | {triplet['target']}"
            )
            transformed_data.append({
                "Instruction": instruction,
                "Question": question,
                "Answer": answer
            })
    
    # Process instruction QA dataset (assumed already in the expected instruction-question-answer format)
    for sample in instruction_data:
        transformed_data.append({
            "Instruction": sample["Instruction"],
            "Question": sample["Question"],
            "Answer": sample["Answer"]
        })
        
    # Build a Hugging Face Dataset from the merged data.
    return Dataset.from_dict({
        "Instruction": [d["Instruction"] for d in transformed_data],
        "Question": [d["Question"] for d in transformed_data],
        "Answer": [d["Answer"] for d in transformed_data]
    })

# --- Load the Dataset ---

enriched_data_path = "enriched_data.json"
instruction_qa_path = "instruction_qa_dataset.json"
dataset = load_dataset(enriched_data_path, instruction_qa_path)

# Create train/test splits (90/10 split here)
dataset = DatasetDict({
    "train": dataset.select(range(int(0.9 * len(dataset)))),
    "test": dataset.select(range(int(0.9 * len(dataset)), len(dataset)))
})

# --- Quantized Model Loading Option ---
# Set up BitsAndBytes configuration for 4-bit quantization.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # "nf4" or "fp4" are available options
    bnb_4bit_compute_dtype="float16"  # or torch.float16 if you prefer
)

# Load the tokenizer as usual.
tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-3-8b")

# Load the model in a quantized mode.
model = LlamaForCausalLM.from_pretrained(
    "hf-internal-testing/llama-3-8b",
    quantization_config=quant_config,
    device_map="auto"
)

# --- Tokenization Function ---
def tokenize_function(example):
    prompt = (
        f"### Instruction: {example['Instruction']}\n"
        f"### Question: {example['Question']}\n"
        f"### Answer: {example['Answer']}"
    )
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Instruction", "Question", "Answer"])

# --- Setup LoRA (PEFT) ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture.
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./llama-3-8b-finetuned",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    deepspeed="zero3_config.json",
    save_total_limit=2,
    push_to_hub=False,
)

# --- Define Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# --- Train the Model ---
trainer.train()

# --- Save the Fine-Tuned Model and Tokenizer ---
trainer.save_model("./llama-3-8b-finetuned")
tokenizer.save_pretrained("./llama-3-8b-finetuned")

