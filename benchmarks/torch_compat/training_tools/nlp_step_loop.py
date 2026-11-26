# Adapted from the pinned Accelerate NLP example, licensed under Apache-2.0.
"""Step-equivalent BERT loop for explicitly comparing eager and compiled steps.

Use the pinned example's model, data, optimizer and scheduler settings. Keep
observation outside the compiled callable. This is not the unmodified official
example; bench_nlp records the selected execution mode in every result.
"""

import time


def training_step(model, optimizer, accelerator, parameters, parity, *, snapshot_buffers=None):
    """Return readouts, or write supplied buffers consumed before the next call."""
    if snapshot_buffers is not None and (not parity or len(snapshot_buffers) != len(parameters)):
        raise ValueError('snapshot buffers require parity and one buffer per parameter')
    if snapshot_buffers is not None and any(buffer.requires_grad for buffer in snapshot_buffers):
        raise ValueError('snapshot buffers must not require gradients')
    def step(**batch):
        outputs = model(**batch)
        loss = outputs.loss / 1  # Official example's accumulation factor.
        accelerator.backward(loss)
        if snapshot_buffers is None:
            gradients = tuple(p.grad.detach().clone() for p in parameters) if parity else ()
        else:
            for buffer, parameter in zip(snapshot_buffers, parameters):
                buffer.copy_(parameter.grad.detach())
            # Returning these tensors would make replay protect them with
            # another full copy. The caller reads the persistent buffers
            # immediately, before the next step can overwrite them.
            gradients = ()
        logits = outputs.logits.detach().clone() if parity else ()
        optimizer.step()
        optimizer.zero_grad()
        return loss.detach(), gradients, logits
    return step


def train_steps(namespace, config, args, execution, parity, finish, out):
    import torch
    import numpy as np
    from safetensors.numpy import save_file
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed

    options = {}
    if getattr(args, 'ddp_bucket_cap_mb', None) is not None:
        from accelerate.utils import DistributedDataParallelKwargs
        options['kwargs_handlers'] = [DistributedDataParallelKwargs(bucket_cap_mb=args.ddp_bucket_cap_mb)]
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, **options)
    set_seed(config['seed'])
    train, evaluation = namespace['training_function'].__globals__['get_dataloaders'](accelerator, config['batch_size'])
    model = AutoModelForSequenceClassification.from_pretrained(
        getattr(args, 'model', 'bert-base-cased'), return_dict=True).to(accelerator.device)
    optimizer_options = {}
    lr = config['lr']
    if args.tensor_learning_rate:
        lr = torch.tensor(lr, device=accelerator.device)
        optimizer_options['capturable'] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, **optimizer_options)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                               num_training_steps=len(train) * config['num_epochs'])
    model, optimizer, train, evaluation, scheduler = accelerator.prepare(model, optimizer, train, evaluation, scheduler)
    named_parameters = tuple(model.named_parameters())
    runners = {}
    snapshot_buffers = None
    if parity and getattr(args, 'parity_shared_snapshots', False):
        with torch.no_grad():
            snapshot_buffers = tuple(torch.empty_like(p, requires_grad=False) for _, p in named_parameters)
            for buffer in snapshot_buffers:
                buffer.zero_()
                if hasattr(buffer, 'sync'):
                    buffer.sync()
        torch.cuda.synchronize()
    step = training_step(model, optimizer, accelerator,
                         tuple(p for _, p in named_parameters), parity, snapshot_buffers=snapshot_buffers)

    step_number = 0
    epoch_times = []
    for epoch in range(config['num_epochs']):
        model.train()
        train_start = time.perf_counter()
        for batch in train:
            signature = tuple((key, tuple(value.shape), str(value.dtype)) for key, value in sorted(batch.items()))
            if signature not in runners:
                runners[signature] = (torch.compile(step, mode='reduce-overhead', dynamic=False)
                                      if execution == 'compiled-step' else step)
            if execution == 'compiled-step':
                torch.compiler.cudagraph_mark_step_begin()
            loss, gradients, logits = runners[signature](**batch)
            if snapshot_buffers is not None:
                gradients = snapshot_buffers
            scheduler.step()
            finish(loss, optimizer.param_groups[0]['lr'])
            if parity:
                save_file({'logits': np.ascontiguousarray(logits.detach().cpu().numpy()),
                           'labels': np.ascontiguousarray(batch['labels'].detach().cpu().numpy())},
                          str(out / ('logits-%d.safetensors' % step_number)))
                save_file({key: np.ascontiguousarray(gradient.detach().cpu().numpy())
                           for (key, _), gradient in zip(named_parameters, gradients)},
                          str(out / ('grad-%d.safetensors' % step_number)))
                save_file({key: np.ascontiguousarray(parameter.detach().cpu().numpy())
                           for key, parameter in named_parameters},
                          str(out / ('param-%d.safetensors' % step_number)))
            step_number += 1
        # Keep the same evaluation transitions, outside training-step timings.
        eval_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for batch in evaluation:
                model(**batch)
        torch.cuda.synchronize()
        wait_start = time.perf_counter()
        # A fast rank must not time the slow rank's evaluation as its first
        # training step, especially when the eval shape compiles once.
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        epoch_times.append(dict(epoch=epoch, training_wall_seconds=eval_start-train_start,
                                eval_seconds=wait_start-eval_start,
                                eval_rank_wait_seconds=time.perf_counter()-wait_start))
    accelerator.end_training()
    captures = []
    for signature, runner in runners.items():
        if hasattr(runner, 'stats'):
            capture = dict(signature=str(signature), stats=runner.stats,
                           refused=runner.refused, graph_refused=runner._graph_refused)
            captures.append(capture)
            if execution == 'compiled-step' and step_number >= 18:
                assert runner.refused is None, capture
                assert runner.stats['graph'] > 0, capture
    return captures, epoch_times
