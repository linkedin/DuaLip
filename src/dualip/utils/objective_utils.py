import torch


def calc_grad(
    dual_grad: torch.Tensor,
    dual_obj: torch.Tensor,
    dual_val: torch.Tensor,
    b_vec: torch.Tensor,
    reg_penalty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dual_grad = dual_grad - b_vec
    dual_obj = dual_obj + reg_penalty + torch.dot(dual_val, dual_grad)
    return dual_grad, dual_obj
