import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Union

import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainerCallback, Trainer, TrainingArguments, \
    DataCollatorForSeq2Seq

from src.utils.defaults import DEVICE, DATA_DIR, DEEP_IMPACT_DIR

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, *args, **kwargs):
        self.profiler.step()


class FineTuner:
    def __init__(self, checkpoint_dir: Union[str, Path], dataset_path: Union[str, Path], output_dir: Union[str, Path],
                 lr: float, epochs: int, batch_size: int, gradient_accumulation_steps: int,
                 logging_dir: Union[str, Path], logging_steps: int, save_steps: int, save_total_limit: int,
                 enable_profiler: bool, seed: int):
        self.enable_profiler = enable_profiler
        self.output_dir = Path(output_dir)

        self.tokenizer = LlamaTokenizer.from_pretrained(checkpoint_dir)
        self.tokenizer.pad_token_id = 0  # making it different from the eos token
        self.model = LlamaForCausalLM.from_pretrained(
            checkpoint_dir,
            load_in_8bit=True,
            device_map=DEVICE,
            torch_dtype=torch.float16
        )
        self.model.train()

        self.lora_config = self.create_peft_config()
        self.dataset = self._load_dataset(dataset_path)

        self.profiler, self.profiler_callback = self._setup_profiler()
        self.training_args = TrainingArguments(
            optim='adamw_torch_fused',
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            fp16=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            num_train_epochs=epochs,
            logging_dir=str(logging_dir),
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            seed=seed,
        )

    def _setup_profiler(self):
        if not self.enable_profiler:
            return nullcontext(), None
        wait, warmup, active, repeat = 1, 1, 2, 1
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{self.output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        return profiler, ProfilerCallback(profiler)

    def create_peft_config(self):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=['q_proj', 'v_proj']
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        return peft_config

    def _load_dataset(self, dataset_path: Union[str, Path]):
        def loader():
            with open(dataset_path, encoding='utf-8') as f:
                for line in f:
                    doc, query = line.strip().split('\t')
                    yield self.generate_prompt_and_tokenize(doc, query)

        return Dataset.from_generator(loader)

    def generate_prompt_and_tokenize(self, document: str, query: str = ''):
        prompt = f'Predict possible search queries for the following document:\n{document}\n---\n'
        if query:
            prompt += query + self.tokenizer.eos_token

        encoded = self.tokenizer(prompt)

        # shifting by 1 happens inside the model
        encoded['labels'] = encoded['input_ids'].copy()
        return encoded

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

            old_state_dict = self.model.state_dict
            self.model.state_dict = (
                lambda x, *_, **__: get_peft_model_state_dict(
                    x, old_state_dict()
                )
            ).__get__(self.model, type(self.model))

            trainer.train()
            self.model.save_pretrained(self.output_dir)


if __name__ == "__main__":
    DIR = DATA_DIR / 'llama2' / 'models_hf'
    parser = argparse.ArgumentParser(description='Finetune Llama2 on Document-Query pairs.')
    parser.add_argument('--checkpoint_dir', type=Path, default=DIR / '7B')
    parser.add_argument('--dataset_path', type=Path, default=DEEP_IMPACT_DIR / 'document-query-pairs.train.tsv')
    parser.add_argument('--output_dir', type=Path, default=DIR / '7B_finetuned')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--logging_dir', type=Path, default=DIR / 'logs')
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--enable_profiler', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    arguments = parser.parse_args()

    fine_tuner = FineTuner(**vars(arguments))
    fine_tuner.train()
