from pathlib import Path
from typing import Union

import torch
from torch.nn import DataParallel
from tqdm.auto import tqdm

from src.deep_impact.models import DeepImpactCrossEncoder
from src.utils.datasets import TopKDataset, Collection, RunFile


class CrossEncoderReRanker:
    def __init__(
            self,
            checkpoint_path: Union[str, Path],
            top_k_path: Union[str, Path],
            collection_path: Union[str, Path],
            output_path: Union[str, Path],
            batch_size: int = 32,
    ):
        self.device = torch.device("cuda")
        self.top_k = TopKDataset(top_k_path=top_k_path)
        self.collection = Collection(collection_path=collection_path)
        self.batch_size = batch_size

        self.model = DeepImpactCrossEncoder.load(checkpoint_path=checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.model = DataParallel(self.model)

        self.run_file = RunFile(run_file_path=output_path)

    def run(self):
        with tqdm(total=len(self.top_k)) as pbar:
            for qid in self.top_k.keys():
                for rank, (pid, score) in enumerate(self.rerank(qid, pbar=pbar), start=1):
                    self.run_file.write(qid, pid, rank, score)
                pbar.update(1)

    @torch.no_grad()
    def rerank(self, qid: int, pbar):
        query = self.top_k.queries[qid]
        top_k_pids = self.top_k[qid]

        batch_encoded_list = []
        scores = []
        for i, pid in enumerate(top_k_pids):
            pbar.set_description(f"Reranking query {qid} ({i + 1}/{len(top_k_pids)})")

            document = self.collection[pid]
            batch_encoded_list.append(DeepImpactCrossEncoder.process_cross_encoder_document_and_query(document, query))

            if len(batch_encoded_list) == self.batch_size or i == len(top_k_pids) - 1:
                input_ids = torch.tensor([x.ids for x in batch_encoded_list], dtype=torch.long)
                attention_mask = torch.tensor([x.attention_mask for x in batch_encoded_list], dtype=torch.long)
                type_ids = torch.tensor([x.type_ids for x in batch_encoded_list], dtype=torch.long)

                scores.extend(self.model(input_ids=input_ids, attention_mask=attention_mask,
                                         token_type_ids=type_ids).squeeze().cpu().tolist())
                batch_encoded_list = []

        return sorted(zip(top_k_pids, scores), key=lambda x: x[1], reverse=True)
