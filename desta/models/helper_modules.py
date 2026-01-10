import torch
import torch.nn as nn

class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
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
    Discriminator to predict transcription embedding from audio tokens.
    Used for adversarial erasure of linguistic content.
    """
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size if output_size is None else output_size)
        )
        
    def forward(self, x):
        return self.net(x)
