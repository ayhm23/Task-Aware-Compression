"""
compression/distillation.py
─────────────────────────────────────────────────────────────────────────────
Knowledge Distillation Compression.

Idea:
    Teacher = full 768-dim encoder (frozen, already computed embeddings)
    Student = lightweight MLP that outputs output_dim embeddings

    The student is trained to:
      1. Match the teacher's output distribution  (KL divergence loss)
      2. Perform well on the downstream task      (task loss)

    Combined loss = α × task_loss + (1-α) × distillation_loss

Why distillation?
    Unlike linear/AE which compress post-hoc, distillation trains a
    NEW encoder that directly produces compact representations while
    preserving the teacher's behaviour — the most "task-aware" approach.

In our setup (embeddings pre-computed):
    We treat the 768-dim cached embeddings as teacher outputs,
    and train a student MLP: 768 → output_dim that mimics them
    while being supervised by the task label.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.base import BaseCompressor
from config import EMBEDDING_DIM, DISTILL_TEMPERATURE, DISTILL_ALPHA


class StudentCompressor(BaseCompressor):
    """
    Student network for knowledge distillation.

    Takes the full 768-dim teacher embedding as input,
    outputs a compact output_dim representation.

    Architecture: 768 → 512 → 256 → output_dim  (with residual-style skip)
    """

    def __init__(
        self,
        input_dim:   int   = EMBEDDING_DIM,
        output_dim:  int   = 128,
        dropout:     float = 0.1,
    ):
        super().__init__(input_dim, output_dim)

        self.student = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Projection head: maps student output back to teacher dim
        # Used to compute distillation loss in teacher's space
        self.proj_to_teacher = nn.Linear(output_dim, input_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress teacher embedding to student embedding.

        Args:
            x : teacher embeddings (batch_size, input_dim=768)
        Returns:
            z : student embeddings (batch_size, output_dim)
        """
        return self.student(x)

    def project_to_teacher_space(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project student output back to teacher's dimensionality.
        Used for computing MSE distillation loss.
        """
        return self.proj_to_teacher(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distillation-style reconstruction loss for task-agnostic training.
        Minimises MSE between the student's projection back to teacher space
        and the original teacher embedding x.

        Args:
            x : teacher embeddings (batch_size, input_dim=768)
        Returns:
            Scalar loss value
        """
        z    = self.forward(x)
        proj = self.project_to_teacher_space(z)
        return F.mse_loss(proj, x)


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation training.

    total_loss = α × task_loss  +  (1 - α) × distillation_loss

    Distillation loss options:
      - 'mse'  : MSE between student projection and teacher embedding
      - 'cosine': cosine similarity loss (direction preservation)
      - 'kl'   : KL divergence between softened distributions
    """

    def __init__(
        self,
        alpha:       float = DISTILL_ALPHA,
        temperature: float = DISTILL_TEMPERATURE,
        dist_type:   str   = "mse",
    ):
        super().__init__()
        self.alpha       = alpha        # weight for task loss
        self.temperature = temperature
        self.dist_type   = dist_type

    def distillation_loss(
        self,
        student_proj: torch.Tensor,   # student projected to teacher space
        teacher_emb:  torch.Tensor,   # original teacher embedding
    ) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher.
        """
        if self.dist_type == "mse":
            return F.mse_loss(student_proj, teacher_emb)

        elif self.dist_type == "cosine":
            # 1 - cosine_similarity (we want to maximise similarity)
            cos_sim = F.cosine_similarity(student_proj, teacher_emb, dim=-1)
            return (1 - cos_sim).mean()

        elif self.dist_type == "kl":
            # Soften with temperature and compute KL
            s = F.log_softmax(student_proj / self.temperature, dim=-1)
            t = F.softmax(teacher_emb    / self.temperature, dim=-1)
            return F.kl_div(s, t, reduction="batchmean") * (self.temperature ** 2)

        else:
            raise ValueError(f"Unknown dist_type: {self.dist_type}")

    def forward(
        self,
        task_loss:    torch.Tensor,
        student_proj: torch.Tensor,
        teacher_emb:  torch.Tensor,
    ) -> tuple:
        """
        Args:
            task_loss    : loss from downstream task head
            student_proj : student output projected to teacher space
            teacher_emb  : original teacher (768-dim) embeddings

        Returns:
            (total_loss, task_loss, dist_loss)
        """
        dist_loss  = self.distillation_loss(student_proj, teacher_emb)
        total_loss = self.alpha * task_loss + (1 - self.alpha) * dist_loss
        return total_loss, task_loss, dist_loss


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch_size   = 32
    teacher_embs = torch.randn(batch_size, 768)  # simulated teacher embeddings

    print("── Student Compressor ──")
    for dim in [32, 64, 128, 256]:
        student = StudentCompressor(input_dim=768, output_dim=dim)
        z       = student(teacher_embs)
        proj    = student.project_to_teacher_space(z)
        print(f"  dim={dim:>3}  z={tuple(z.shape)}  "
              f"proj={tuple(proj.shape)}  {student}")

    print("\n── Distillation Loss ──")
    student = StudentCompressor(input_dim=768, output_dim=64)
    z       = student(teacher_embs)
    proj    = student.project_to_teacher_space(z)

    # Fake task loss
    task_loss = torch.tensor(0.5)

    for dist_type in ["mse", "cosine"]:
        loss_fn = DistillationLoss(alpha=0.7, dist_type=dist_type)
        total, tl, dl = loss_fn(task_loss, proj, teacher_embs)
        print(f"  [{dist_type}] total={total.item():.4f}  "
              f"task={tl.item():.4f}  distill={dl.item():.4f}")
