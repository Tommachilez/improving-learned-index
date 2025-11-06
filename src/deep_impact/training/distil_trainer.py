import torch

from .trainer import Trainer


class DistilMarginMSE:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def __call__(self, output, target):
        """
        Calculates the MSE loss between the teacher and student scores
        :param output: Shape (batch_size, n) with positive and negative predicted scores
        :param target: Shape (batch_size, n) with positive and negative teacher scores
        :return: MSE loss
        """
        student_positive_scores = output[:, 0]
        student_negative_scores = output[:, 1:]
        student_margin = student_positive_scores.unsqueeze(1) - student_negative_scores

        # Calculate margin for teacher
        teacher_positive_scores = target[:, 0]
        teacher_negative_scores = target[:, 1:]
        teacher_margin = teacher_positive_scores.unsqueeze(1) - teacher_negative_scores

        return self.loss(student_margin, teacher_margin)


class DistilKLLoss:
    """Distillation loss from: Distilling Dense Representations for Ranking using Tightly-Coupled Teachers
    link: https://arxiv.org/abs/2010.11386
    """

    def __init__(self):
        self.loss = torch.nn.KLDivLoss(reduction="none")

    def __call__(self, output, target):
        # student_scores = torch.log_softmax(output, dim=1)
        # teacher_scores = torch.softmax(target, dim=1)
        # return self.loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)

        # Check the dimensionality of the input tensor
        if output.dim() == 1:
            # 1D Input (e.g., shape [5])
            # We apply softmax along the only dimension (dim=0).
            dim = 0
            # For a 1D tensor, we sum all losses and take the mean.
            reduction = lambda x: x.sum().mean()

        elif output.dim() == 2:
            # 2D Input (e.g., shape [2, 3])
            # We apply softmax across the scores (dim=1).
            dim = 1
            # We sum the KL loss for each item in the batch (dim=1),
            # then average those sums (dim=0).
            reduction = lambda x: x.sum(dim=dim).mean(dim=0)

        else:
            raise ValueError(
                f"DistilKLLoss expects 1D or 2D input, but got {output.dim()}D tensor."
            )

        student_scores = torch.log_softmax(output, dim=dim)
        teacher_scores = torch.softmax(target, dim=dim)

        # Apply the KLDivLoss
        loss_tensor = self.loss(student_scores, teacher_scores)

        # Apply the correct reduction based on the input shape
        return reduction(loss_tensor)


class DistilTrainer(Trainer):
    loss = DistilKLLoss()

    def get_output_scores(self, batch):
        input_ids, attention_mask, type_ids = self.get_input_tensors(batch['encoded_list'])
        document_term_scores = self.model(input_ids, attention_mask, type_ids)

        masks = batch['masks'].to(self.gpu_id)
        # Return the flat list of scores, which is what the loss function expects
        return (masks * document_term_scores).sum(dim=1).squeeze(-1)

    def evaluate_loss(self, outputs, batch):
        # # distillation loss
        # teacher_scores = batch['scores'].view(self.batch_size, -1).to(self.gpu_id)
        # We need the teacher_scores tensor to have the same flat shape as the 'outputs' tensor.
        teacher_scores = batch['scores'].to(self.gpu_id)
        return self.loss(outputs, teacher_scores)
