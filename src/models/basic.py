import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.optim import AdamW, lr_scheduler


def logit_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate how much classes in logits were correctly
    predicted.

    Args:
        logits: output of the model
        target: target class indices

    Returns:

    """
    idx = logits.max(1).indices
    acc = (idx == target).int()
    return acc.sum() / torch.numel(acc)


class ImageEmbedding(nn.Module):

    def __init__(self, size: int, hidden_size: int, num_patches: int, dropout: float = 0.2):
        """Preprocess patchified image into the inner model tensor

        Args:
            size: context size of input
            hidden_size: hidden size of tensor, i.e. new context size of tensor
            num_patches: amount of image patches
            dropout: dropout coefficient
        """
        super().__init__()

        # Input context size is too small for inner context vector
        # Should be expanded.
        self.projection = nn.Linear(size, hidden_size)
        self.class_token = nn.Parameter(torch.rand(1, hidden_size))
        self.position = nn.Parameter(torch.rand(1, num_patches + 1, hidden_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, inp) -> torch.Tensor:
        """Create linear projection from input image,
        prepend class token tensor, and add position embedding

        Args:
            inp: batch of patchified images

        Returns:
            Inner model tensor
        """
        res = self.projection(inp)

        # Repeat on a batch size
        class_token = self.class_token.repeat(res.size(0), 1, 1)  # batch_size x 1 x output_size
        res = torch.concat([class_token, res], dim=1)

        position = self.position.repeat(res.size(0), 1, 1)
        return self.dropout(res + position)


class AttentionHead(nn.Module):

    def __init__(self, size: int):
        """Calculate and apply attention among
        image patches

        Args:
            size: hidden size
        """
        super(AttentionHead, self).__init__()

        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)

    def forward(self, input_tensor) -> torch.Tensor:
        """Calculate attention

        Args:
            input_tensor: inner model tensor

        Returns:
            Tensor with attention scores applied
        """
        q, k, v = self.query(input_tensor), self.key(input_tensor), self.value(input_tensor)

        scale = q.size(1) ** 0.5
        scores = torch.bmm(q, k.transpose(1, 2)) / scale

        scores = F.softmax(scores, dim=-1)

        # 8 x 64 x 64 @ 8 x 64 x 48 = 8 x 64 x 48
        output = torch.bmm(scores, v)
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, size: int, num_heads: int):
        """Unite several attention heads together
        and pass through linear projection

        Args:
            size: hidden size
            num_heads: number of attention heads
        """
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(size) for _ in range(num_heads)])
        self.linear = nn.Linear(size * num_heads, size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run all attention heads for input tensor

        Args:
            input_tensor: batches of inner model images
        Returns:
            Tensor with attention heads applied
        """
        s = [head(input_tensor) for head in self.heads]
        s = torch.cat(s, dim=-1)

        output = self.linear(s)
        return output


class Encoder(nn.Module):

    def __init__(self, size: int, num_heads: int, dropout: float = 0.1):
        """Standard transformer encoder with Multi Head Attention
        and Feed Forward Network
        Args:
            size: hidden size
            num_heads: number of heads
            dropout: dropout coefficient
        """
        super().__init__()

        self.attention = MultiHeadAttention(size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout)
        )
        self.norm_attention = nn.LayerNorm(size)
        self.norm_feed_forward = nn.LayerNorm(size)

    def forward(self, input_tensor) -> torch.Tensor:
        attn = input_tensor + self.attention(self.norm_attention(input_tensor))
        output = attn + self.feed_forward(self.norm_feed_forward(attn))
        return output


class ViT(pl.LightningModule):

    def __init__(self, size: int, hidden_size: int, num_patches: int, num_classes: int, num_heads: int,
                 num_encoders: int, emb_dropout: float = 0.1, dropout: float = 0.1,
                 lr: float = 1e-4, min_lr: float = 4e-5,
                 weight_decay: float = 0.1, epochs: int = 200):
        """The main module. Unites the ViT model with training functionality from
        pytorch_lightning

        Args:
            size: context size of input tensor
            hidden_size: hidden size
            num_patches: number of patches
            num_classes: number of classes
            num_heads: number of attention heads
            num_encoders: number of encoders
            emb_dropout: dropout coefficient for InputEmbedding module
            dropout: dropout coefficient for Encoder module
            lr: learning rate coefficient
            min_lr: minimum value of learning rate a scheduler can set
            weight_decay: weight decay coefficient
            epochs: max number of epochs
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.embedding = ImageEmbedding(size, hidden_size, num_patches, dropout=emb_dropout)

        self.encoders = nn.Sequential(
            *[Encoder(hidden_size, num_heads, dropout=dropout) for _ in range(num_encoders)],
        )
        self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_tensor)
        attn = self.encoders(emb)

        return self.mlp_head(attn[:, 0, :])

    def training_step(self, batch, batch_idx):
        """The code to run in one forward training step.
        The remaining operations are done by pytorch_lightning internally.

        Args:
            batch: tuple of input and target
            batch_idx: index of batch

        Returns:
            Loss
        """
        input_batch, target = batch

        logits = self(input_batch)
        loss = F.cross_entropy(logits, target)

        if batch_idx % 5 == 0:
            self.log("train_acc", logit_accuracy(logits, target), prog_bar=True)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """The same as training step but for validation.

        Args:
            batch: tuple of input and target
            batch_idx: index of batch

        Returns:
            Validation loss
        """
        input_batch, target = batch
        output = self(input_batch)

        loss = F.cross_entropy(output, target)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", logit_accuracy(output, target), prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler
        Read more at https://lightning.ai/docs/pytorch/latest/common/optimization.html#automatic-optimization

        Returns:
            Optimizers
        """
        optimizer = AdamW(self.configure_parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=self.min_lr)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_parameters(self):
        """LayerNorm has its own regularization techniques. We should exclude
        LayerNorm parameters from weight decay.

        This code is mostly taken from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

        Returns:
            List of parameters with their optimizer options
        """
        no_decay_modules = (nn.LayerNorm,)
        decay_modules = (nn.Linear,)

        decay = set()
        no_decay = set()

        # this, of course, makes training (loss reduction) slower
        for module_name, module in self.named_modules():
            if module is self:
                continue
            for param_name, value in module.named_parameters():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith('bias'):
                    no_decay.add(full_name)
                elif param_name.endswith('weight') and isinstance(module, no_decay_modules):
                    no_decay.add(full_name)
                elif param_name.endswith('weight') and isinstance(module, decay_modules):
                    decay.add(full_name)

        optim_groups = [
            {"params": [v for name, v in self.named_parameters() if name in decay],
             "weight_decay": self.weight_decay},
            {"params": [v for name, v in self.named_parameters() if name in no_decay],
             "weight_decay": 0}
        ]
        return optim_groups
