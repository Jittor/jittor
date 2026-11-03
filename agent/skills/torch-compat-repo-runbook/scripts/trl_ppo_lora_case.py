"""Official PPO constructor using three FP32 bases and PEFT reference context."""


def build(args, torch, tokenizer, models, record):
    del record
    from datasets import Dataset
    from trl import PPOConfig, PPOTrainer
    from trl_dpo_case import records

    policy = models('policy')
    reward = models('reward', kind='sequence_classification')
    reward.eval()
    value = models('value', kind='sequence_classification')
    rows = [{'input_ids': tokenizer(row['prompt'], add_special_tokens=False)['input_ids']}
            for row in records()]
    data = Dataset.from_list(rows)
    settings = dict(args.training_kwargs)
    settings.update(per_device_train_batch_size=2, per_device_eval_batch_size=2,
                    response_length=4, local_rollout_forward_batch_size=2,
                    num_sample_generations=0, num_ppo_epochs=1, num_mini_batches=1 if args.steps == 1 else 2,
                    temperature=1., whiten_rewards=False, missing_eos_penalty=None,
                    total_episodes=2 if args.steps == 1 else 10, dataset_num_proc=None)
    trainer = PPOTrainer(args=PPOConfig(**settings), processing_class=tokenizer,
                         model=policy, ref_model=None, reward_model=reward,
                         value_model=value, train_dataset=data, eval_dataset=data)
    if trainer.ref_model is not None or not trainer.is_peft_model:
        raise AssertionError('Require official adapter-disabled policy reference')
    loader = torch.utils.data.DataLoader(data, batch_size=trainer.local_dataloader_batch_size,
                                        shuffle=False, collate_fn=trainer.data_collator, drop_last=True)
    trainer.dataloader = trainer.accelerator.prepare(loader)
    metadata = dict(algorithm='ppo_lora', model_scope='full pretrained Qwen3-0.6B, three FP32 bases',
                    roles='policy/value LoRA and value score trained; all other weights frozen',
                    reference='official PPOTrainer null_ref_context disables policy adapters',
                    expected_optimizer_steps=args.steps, unique_prompts=10,
                    rollout_batches=trainer.args.num_total_batches,
                    local_rollout_batch_size=trainer.args.local_batch_size,
                    mini_batches_per_rollout=trainer.args.num_mini_batches,
                    prompts_per_rollout_batch=trainer.local_dataloader_batch_size,
                    total_episodes=trainer.args.total_episodes,
                    max_new_tokens=4, data_order='sequential; official PPO NumPy shuffle retained',
                    scalar_loss='unchanged official loss observed at accelerator.backward',
                    upstream_trainer=PPOTrainer.__module__+'.'+PPOTrainer.__name__)
    if metadata['rollout_batches'] != (1 if args.steps == 1 else 5) or metadata['local_rollout_batch_size'] != 2:
        raise AssertionError('Incorrect two-prompt rollout count for the selected update protocol')
    return trainer, metadata
