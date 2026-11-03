"""Official Trainer configurations; only ordering and rollout inputs are controlled."""
import json

from trl_dpo_case import records as preference_records
from trl_grpo_case import records as prompt_records, answer_reward


TEXTS = (
    'Two plus three equals five.', 'Nine minus four equals five.',
    'Three times four equals twelve.', 'Eight divided by two equals four.',
    'The capital of France is Paris.', 'The capital of Japan is Tokyo.',
    'A week has seven days.', 'A triangle has three sides.',
    'Ten is larger than two.', 'The opposite of hot is cold.',
)


def case_records(algorithm, max_length):
    context = ('Read these facts carefully. A week has seven days. A triangle has three sides. ' * 12
               if max_length == 128 else '')
    if algorithm == 'sft':
        short_context = 'Read these facts. A week has seven days. A triangle has three sides. ' * 2
        return [{'text': (context if max_length != 128 or index % 2 == 0 else short_context) + text}
                for index, text in enumerate(TEXTS)]
    rows = preference_records() if algorithm == 'dpo' else prompt_records()
    return [dict(row, prompt=context + row['prompt']) for row in rows]


def canonicalize(torch, prepared):
    """Preserve the official mean objective while fixing backend shuffle order."""
    prompts = prepared['prompt_ids'].detach().cpu().tolist()
    completions = prepared['completion_ids'].detach().cpu().tolist()
    count = len(prompts)
    order = sorted(range(count), key=lambda index: (prompts[index], completions[index]))
    indices = torch.tensor(order, dtype=torch.long, device=prepared['prompt_ids'].device)
    for name, value in list(prepared.items()):
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            prepared[name] = value[indices]
        elif isinstance(value, (tuple, list)) and len(value) == count:
            prepared[name] = type(value)(value[index] for index in order)
    return prepared


def attach_rollout_control(args, torch, tokenizer, trainer, manifest, record):
    """Oracle samples; candidate replays only those discrete generated tokens."""
    calls = []
    if args.runtime == 'jittor':
        if args.reference is None:
            raise ValueError('GRPO candidate requires the independent oracle trace')
        reference = json.loads((args.reference / 'manifest.json').read_text())
        if reference['status'] != 'complete' or reference['runtime'] != 'torch':
            raise ValueError('Require a completed genuine Torch rollout producer')
        if reference['protocol'] != manifest['protocol']:
            raise ValueError('Rollout protocol mismatch')
        calls = [call for call in reference['generation_calls']
                 if not args.resume or call['before_global_step'] >= 5]
    original = trainer.model.generate

    def generate(*positional, **kwargs):
        if positional:
            raise AssertionError('Expected keyword-only official generate call')
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        inputs = dict(input_ids=ids.detach().cpu().tolist(), attention_mask=mask.detach().cpu().tolist())
        config, _ = trainer.model._prepare_generation_config(
            kwargs['generation_config'], use_model_defaults=False,
            **{key: value for key, value in kwargs.items() if key != 'generation_config'})
        configuration = config.to_dict()
        index = len(manifest['generation_calls'])
        if args.runtime == 'torch':
            output = original(use_model_defaults=False, **kwargs)
        else:
            if index >= len(calls) or calls[index]['inputs'] != inputs or calls[index]['config'] != configuration:
                raise AssertionError('Generated-token replay inputs/configuration differ')
            output = torch.tensor(calls[index]['output_ids'], dtype=ids.dtype, device=ids.device)
        output_ids = output.detach().cpu().tolist()
        if len(output_ids) != len(inputs['input_ids']) or any(
                row[:len(prompt)] != prompt for row, prompt in zip(output_ids, inputs['input_ids'])):
            raise AssertionError('Generated output changes prompt prefix')
        before_step = int(trainer.state.global_step)
        same_step = sum(call['before_global_step'] == before_step for call in manifest['generation_calls'])
        manifest['generation_calls'].append(dict(inputs=inputs, config=configuration, output_ids=output_ids, before_global_step=before_step))
        prefix = 'step/%d/generation/%d/' % (before_step+1, same_step)
        record(prefix+'input_ids', ids, exact=True)
        record(prefix+'attention_mask', mask, exact=True)
        record(prefix+'output_ids', output, exact=True)
        return output
    trainer.model.generate = generate


def build_trainer(args, torch, model, tokenizer, observations, callback, manifest, record):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer

    settings = dict(output_dir=str(args.output / 'trainer'), use_cpu=args.device == 'cpu',
        bf16=args.precision == 'bf16', fp16=args.precision == 'fp16', tf32=False,
        max_steps=10, per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=1e-4, weight_decay=.01, optim='adamw_torch', adam_beta1=.9, adam_beta2=.999,
        adam_epsilon=1e-8, lr_scheduler_type='linear', warmup_steps=0, max_grad_norm=0.,
        gradient_checkpointing=False, dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, logging_steps=1, disable_tqdm=True, report_to=[],
        save_strategy='steps', save_steps=5, save_total_limit=2, eval_strategy='no',
        seed=0, data_seed=0, save_safetensors=True)
    rows = case_records(args.algorithm, args.max_length)
    common = dict(model=model, train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                  callbacks=[callback])
    if args.algorithm == 'sft':
        class Observed(observations, SFTTrainer):
            pass
        settings.update(max_length=args.max_length, packing=False, dataset_text_field='text')
        return Observed(args=SFTConfig(**settings), **common)
    if args.algorithm == 'dpo':
        class Observed(observations, DPOTrainer):
            pass
        settings.update(max_length=args.max_length, max_prompt_length=args.max_length-8,
            max_completion_length=8, beta=.1, loss_type='sigmoid', disable_dropout=True,
            precompute_ref_log_probs=False, use_liger_loss=False, use_logits_to_keep=False,
            padding_free=False, generate_during_eval=False)
        trainer = Observed(args=DPOConfig(**settings), **common)
        if trainer.ref_model is not None:
            raise AssertionError('LoRA DPO should use disabled policy adapters for its reference')
        return trainer

    class Observed(observations, GRPOTrainer):
        def _prepare_inputs(self, inputs):
            return canonicalize(torch, super()._prepare_inputs(inputs))

    def reward(prompts, completions, target, **kwargs):
        del prompts, kwargs
        return [answer_reward(completion, expected) for completion, expected in zip(completions, target)]

    settings.update(per_device_train_batch_size=2, num_generations=2,
        steps_per_generation=args.gradient_accumulation, num_iterations=1,
        shuffle_dataset=False, max_prompt_length=args.max_length-4, max_completion_length=4,
        beta=.04, loss_type='grpo', scale_rewards=True, temperature=1., top_p=1., top_k=0,
        use_vllm=False, generation_kwargs={'use_cache': False})
    trainer = Observed(args=GRPOConfig(**settings), reward_funcs=reward, **common)
    if trainer.ref_model is not None:
        raise AssertionError('LoRA GRPO should use disabled policy adapters for its reference')
    attach_rollout_control(args, torch, tokenizer, trainer, manifest, record)
    return trainer
