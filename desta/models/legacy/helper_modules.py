import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer used by legacy adversarial-erasure experiments.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -ctx.alpha * grad_output
        return grad_input, None


class TranscriptionDiscriminator(nn.Module):
    """
    Legacy discriminator for predicting transcription embeddings from audio tokens.
    """
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size if output_size is None else output_size),
        )

    def forward(self, x):
        return self.net(x)
