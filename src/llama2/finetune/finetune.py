import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Union

import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, # Changed from LlamaForCausalLM for better compatibility
    AutoTokenizer,      # Changed from LlamaTokenizer
    BitsAndBytesConfig, # Added for 4-bit quantization
    TrainerCallback,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from src.utils.defaults import DEVICE, LLAMA_DIR, LLAMA_HUGGINGFACE_CHECKPOINT, DEEP_IMPACT_DIR

LORA_CONFIG = {
    'task_type': TaskType.CAUSAL_LM,
    'inference_mode': False,
    'r': 16,  # Increased rank for better performance
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    # Target more layers for better fine-tuning
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
}


class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, *args, **kwargs):
        self.profiler.step()


class FineTuner:
    def __init__(self, checkpoint_dir: Union[str, Path], dataset_path: Union[str, Path], output_dir: Union[str, Path],
                 lr: float, epochs: int, batch_size: int, gradient_accumulation_steps: int, save_steps: int,
                 save_total_limit: int, logging_dir: Union[str, Path], logging_steps: int, enable_profiler: bool,
                 seed: int):
        self.enable_profiler = enable_profiler
        self.logging_dir = Path(logging_dir)
        self.output_dir = Path(output_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.tokenizer.pad_token_id = 0  # making it different from the eos token

        self.dataset = self._load_dataset(dataset_path)

        self.model, self.lora_config = self._load_model_and_create_peft_config(checkpoint_dir)

        self.profiler, self.profiler_callback = self._setup_profiler()
        self.training_args = TrainingArguments(
            seed=seed,
            optim='paged_adamw_32bit', # Recommended optimizer for QLoRA
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            bf16=torch.cuda.is_bf16_supported(), # Check for bf16 support
            fp16=not torch.cuda.is_bf16_supported(),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True, # Enable gradient checkpointing for memory savings
            learning_rate=lr,
            # lr_scheduler_type='cosine',
            num_train_epochs=epochs,

            # logging
            logging_strategy='steps',
            logging_dir=str(logging_dir),
            logging_steps=logging_steps,

            # saving
            save_strategy='steps',
            save_steps=save_steps,
            save_total_limit=save_total_limit,
        )

    def _setup_profiler(self):
        directory = self.logging_dir / 'tensorboard'
        if not self.enable_profiler:
            return nullcontext(), None
        wait, warmup, active, repeat = 1, 1, 2, 1
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(directory)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        return profiler, ProfilerCallback(profiler)

    @staticmethod
    def _load_model_and_create_peft_config(checkpoint_dir: Union[str, Path]):
        # --- REVISED: Use 4-bit quantization (QLoRA) ---
        compute_dtype = getattr(torch, "bfloat16")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            quantization_config=bnb_config,
            device_map="auto" # Automatically handle device mapping
        )
        model.train()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, peft_config

    def _load_dataset(self, dataset_path: Union[str, Path]):
        def loader():
            with open(dataset_path, encoding='utf-8') as f:
                for line in f:
                    doc, query = line.strip().split('\t')
                    yield self.generate_prompt_and_tokenize(doc, query)

        return Dataset.from_generator(loader)

    def generate_prompt_and_tokenize(self, document: str, query: str = ''):
        prompt = f'Dự đoán các truy vấn tìm kiếm có thể có cho tài liệu sau đây:\n{document}\n---\n'

        # if query:
        #     prompt += query + self.tokenizer.eos_token

        # encoded = self.tokenizer(prompt)

        # # shifting by 1 happens inside the model
        # encoded['labels'] = encoded['input_ids'].copy()

        # 1. Tokenize prompt and query SEPARATELY.
        #    Do not add special tokens to the prompt, as it's not the end of the sequence.
        prompt_encoding = self.tokenizer(prompt, add_special_tokens=False)

        #    Add the eos token to the query, as it marks the end of the generation.
        query_encoding = self.tokenizer(query + self.tokenizer.eos_token, add_special_tokens=False)

        # 2. Combine them to create the full input and label sequences
        #    We add the BOS token (tokenizer.bos_token_id) because we discard it above.
        input_ids = [self.tokenizer.bos_token_id] + prompt_encoding['input_ids'] + query_encoding['input_ids']

        #    Create labels: -100 for the prompt part (including BOS), and real tokens for the query part.
        labels = ([-100] * (len(prompt_encoding['input_ids']) + 1)) + query_encoding['input_ids']

        # 3. NOW truncate the combined sequences from the RIGHT
        #    This is crucial. We must ensure input_ids and labels are truncated the same way.
        max_length = 2048
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        # 4. Create the attention mask
        attention_mask = [1] * len(input_ids)

        # 5. The tokenizer's __call__ method (used in your original code) would also pad.
        #    We must replicate that padding for the collator. The DataCollator will handle this,
        #    but we must ensure the `return` is a dict.

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def train(self):
        with self.profiler:
            trainer = Trainer(
                model=self.model,
                train_dataset=self.dataset,
                args=self.training_args,
                data_collator=DataCollatorForSeq2Seq(
                    self.tokenizer,
                    pad_to_multiple_of=8,
                    padding=True,
                    return_tensors='pt'
                ),
                callbacks=[self.profiler_callback] if self.enable_profiler else [],
            )
            self.model.config.use_cache = False

            trainer.train()
            self.model.save_pretrained(str(self.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune Llama2 on Document-Query pairs.')
    parser.add_argument('--checkpoint_dir', type=Path, default=LLAMA_HUGGINGFACE_CHECKPOINT)
    parser.add_argument('--dataset_path', type=Path, default=DEEP_IMPACT_DIR / 'document-query-pairs.train.tsv')
    parser.add_argument('--output_dir', type=Path, default=LLAMA_DIR / '7B_finetuned')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--logging_dir', type=Path, default=LLAMA_DIR / 'logs')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--enable_profiler', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    arguments = parser.parse_args()

    fine_tuner = FineTuner(**vars(arguments))
    fine_tuner.train()
