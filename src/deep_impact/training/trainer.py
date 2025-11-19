import os
from pathlib import Path
from typing import Union

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import wandb

from src.utils.checkpoint import ModelCheckpoint
from src.utils.logger import Logger
from src.deep_impact.evaluation.nano_beir_evaluator import BaseEvaluator


class Trainer:
    logger = Logger(Path(__file__).stem, stream=True)

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            checkpoint_dir: Union[str, Path],
            batch_size: int,
            save_every: int,
            save_best: bool = True,
            seed: int = 42,
            gradient_accumulation_steps: int = 1,
            eval_every: int = 500,
            evaluator: BaseEvaluator = None,
            use_wandb: bool = False,
            logging_dir: Union[str, Path] = None,
            logging_steps: int = 10,
            config = None
    ) -> None:
        self.seed = seed
        self.gpu_id = torch.distributed.get_rank()
        self.n_ranks = torch.distributed.get_world_size()
        self.model = model.to(self.gpu_id)
        self.optimizer = optimizer
        self.train_data = train_data
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_every = eval_every
        self.evaluator = evaluator
        self.logging_steps = logging_steps
        self.use_wandb = use_wandb
        self.writer = None

        model_name = self.model.__class__.__name__
        last_checkpoint_path = (checkpoint_dir /
                                f'{model_name}_{ModelCheckpoint.LATEST_SNAPSHOT_SUFFIX}.{ModelCheckpoint.EXTENSION}')
        if os.path.exists(last_checkpoint_path):
            self.checkpoint_callback = ModelCheckpoint.load(
                model=self.model,
                optimizer=self.optimizer,
                last_checkpoint_path=last_checkpoint_path,
                save_every=save_every,
                save_best=save_best,
            )
            if self.checkpoint_callback.batch_size:
                # assume n_ranks info is multiplied in batch_size
                self.checkpoint_callback.step = (self.checkpoint_callback.step * self.checkpoint_callback.batch_size) \
                                                // (self.batch_size * self.n_ranks)
            else:
                self.logger.info(f"Assuming previous training was done on same number of GPUs & batch size.")
        else:
            self.checkpoint_callback = ModelCheckpoint(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint_dir=checkpoint_dir,
                save_every=save_every,
                save_best=save_best,
            )
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_callback.batch_size = self.batch_size * self.n_ranks
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.gpu_id == 0:
            # 1. Initialize TensorBoard Writer
            log_path = Path(logging_dir) if logging_dir else Path("logs/tensorboard")
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))

            # 2. Initialize WandB to sync TensorBoard
            if self.use_wandb:
                wandb.init(
                    project="DeepImpact",
                    config=config,
                    sync_tensorboard=True
                )
                self.logger.info("WandB initialized with TensorBoard sync.")

    def train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.model.train()

        self.criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')

        # Resume training if checkpoint exists i.e. step > 0
        remaining = len(self.train_data) - self.checkpoint_callback.step

        total_dataset_len = len(self.train_data.dataset)

        self.train_data = iter(self.train_data)
        if self.checkpoint_callback.step:
            self.skip()

        with tqdm(total=remaining) as progress_bar:
            train_loss = 0

            for i, batch in enumerate(self.train_data):
                with torch.cuda.amp.autocast():
                    outputs = self.get_output_scores(batch)
                    loss = self.evaluate_loss(outputs, batch)

                    loss /= self.gradient_accumulation_steps

                scaler.scale(loss).backward()
                current_loss = loss.detach().cpu().item()
                train_loss += current_loss

                if i % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                if self.gpu_id == 0:

                    if i % self.logging_steps == 0:
                        step = self.checkpoint_callback.step + (i * self.batch_size * self.n_ranks) # Estimate global step
                        self.writer.add_scalar("train/loss", current_loss, step)
                        self.writer.add_scalar("train/avg_loss", train_loss / (i + 1), step)
                        self.writer.add_scalar("train/epoch", (i * self.batch_size * self.n_ranks) / total_dataset_len, step)
                        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], step)

                    if i % self.eval_every == 0 and self.evaluator is not None:
                        self.logger.info(f"Evaluating NanoBEIR at iteration {i}")
                        metrics = self.evaluator.evaluate_all(self.model.module)
                        self.logger.info(f"Metrics: {metrics}")
                        # Log evaluation metrics to TensorBoard
                        step = self.checkpoint_callback.step + (i * self.batch_size * self.n_ranks)
                        # Assuming metrics structure: {'avg': {'NDCG@10': 0.5, ...}, ...}
                        for dataset, values in metrics.items():
                            if dataset == "avg":
                                for metric_name, val in values.items():
                                    # Flatten dict if necessary or just log key metrics
                                    if isinstance(val, dict):
                                        for k, v in val.items():
                                            self.writer.add_scalar(f"eval/{dataset}/{metric_name}/{k}", v, step)
                                    else:
                                        self.writer.add_scalar(f"eval/{dataset}/{metric_name}", val, step)
                        # write metrics to file as as single line, add iteration number
                        with open(self.checkpoint_dir / "metrics.txt", "a") as f:  
                            f.write(json.dumps({"iteration": i, "metrics": metrics}) + "\n")

                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Average Train Loss: {train_loss / (i + 1) * 100:.4f}, "
                        f"Current Loss: {current_loss * 100:.4f}, "
                        f"Examples Seen: {i * self.batch_size * self.n_ranks}")
                    self.checkpoint_callback()
                            

            self.checkpoint_callback.save('final')

    def get_input_tensors(self, encoded_list):
        input_ids = torch.tensor([x.ids for x in encoded_list], dtype=torch.long).to(self.gpu_id)
        attention_mask = torch.tensor([x.attention_mask for x in encoded_list], dtype=torch.long).to(
            self.gpu_id)
        type_ids = torch.tensor([x.type_ids for x in encoded_list], dtype=torch.long).to(self.gpu_id)
        return input_ids, attention_mask, type_ids

    def get_output_scores(self, batch):
        input_ids, attention_mask, type_ids = self.get_input_tensors(batch['encoded_list'])
        document_term_scores = self.model(input_ids, attention_mask, type_ids)

        masks = batch['masks'].to(self.gpu_id)
        return (masks * document_term_scores).sum(dim=1).squeeze(-1).view(self.batch_size, -1)

    def evaluate_loss(self, outputs, batch):
        labels = torch.zeros(self.batch_size, dtype=torch.long).to(self.gpu_id)
        return self.criterion(outputs, labels)

    def skip(self):
        if self.gpu_id == 0:
            self.logger.info(
                f"Resuming training from step {self.checkpoint_callback.step}. "
                f"Skipping {self.checkpoint_callback.step * self.batch_size * self.n_ranks} seen examples."
            )

        with tqdm(total=self.checkpoint_callback.step) as progress_bar:
            for i, _ in enumerate(self.train_data, start=1):
                if i == self.checkpoint_callback.step:
                    break
                if self.gpu_id == 0:
                    progress_bar.update(1)

    @staticmethod
    def ddp_setup():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(torch.distributed.get_rank())

    @staticmethod
    def ddp_cleanup():
        torch.distributed.destroy_process_group()
