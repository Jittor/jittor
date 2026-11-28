"""Compile equivalent CV training work and preserve OneCycle's live scalars."""


def native_schedule_tensors(optimizer, torch, device):
    """Keep native compiler guards independent of per-step LR/beta values.

    Jittor refreshes these values in its fused optimizer prologue. Native Adam
    accepts Tensor hyperparameters; FP64 retains Python scheduler precision.
    """
    if hasattr(torch, '_torch_compat_install_context'):
        return ()
    handles = []
    for group in optimizer.param_groups:
        lr = torch.tensor(float(group['lr']), device=device, dtype=torch.float64)
        betas = tuple(torch.tensor(float(x), device=device, dtype=torch.float64)
                      for x in group['betas'])
        group['capturable'] = True
        group['lr'], group['betas'] = lr, betas
        handles.append((group, lr, betas))
    return tuple(handles)


def advance_scheduler(scheduler, handles):
    """Apply the original schedule, then refresh the same device holders."""
    scheduler.step()
    for group, lr, betas in handles:
        value = group['lr']
        if value is not lr:
            lr.fill_(value)
        for target, value in zip(betas, group['betas']):
            if value is not target:
                target.fill_(value)
        group['lr'], group['betas'] = lr, betas


def training_step(model, optimizer, accelerator, torch, snapshot=False, return_logits=False):
    """Keep backward/update/zeroing inside the same callable on both stacks."""
    parameters = tuple(p for p in model.parameters() if p.requires_grad)

    def step(inputs, labels):
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accelerator.backward(loss)
        gradients = tuple(p.grad.detach().clone() for p in parameters) if snapshot else ()
        logits = outputs.detach().clone() if return_logits else ()
        optimizer.step()
        optimizer.zero_grad()
        if return_logits:
            return loss.detach(), gradients, logits
        return loss.detach(), gradients

    return step
