import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue

                    gradient = parameter.grad
                    state = self.state[parameter]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)

                    state["step"] += 1
                    step = state["step"]

                    first_moment = state["exp_avg"]
                    second_moment = state["exp_avg_sq"]
                    if weight_decay != 0:
                        parameter.mul_(1 - lr * weight_decay)

                    first_moment.mul_(beta1).add_(
                        gradient,
                        alpha=1 - beta1,
                    )
                    second_moment.mul_(beta2).addcmul_(
                        gradient,
                        gradient,
                        value=1 - beta2,
                    )
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    bias_correction2_sqrt = math.sqrt(bias_correction2)

                    step_size = lr / bias_correction1
                    
                    denominator = (second_moment.sqrt()/ bias_correction2_sqrt).add_(eps)
                    parameter.addcdiv_(
                        first_moment,
                        denominator,
                        value=-step_size
                    )
        return loss