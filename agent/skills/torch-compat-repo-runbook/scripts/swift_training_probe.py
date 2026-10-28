"""Observe the real Swift SFT loader/template/Trainer three-step FP32 protocol.

The only subclass adds public callbacks and forward hooks to SwiftSft.train.
Swift owns dataset encoding, loss, backward, optimizer/scheduler and step order.
Integer token inputs have no requested input gradients; the embedding output
has an additional observed intermediate gradient, explicitly recorded as such.
"""

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import os
from email.parser import BytesParser
from pathlib import Path
import subprocess
import sys
import zipfile

CASE = 'ms_swift_trainer_full_llama'
UPSTREAM_CALLS = {
    'swift.model.register.get_model_processor': 1,
    'swift.model.register.ModelLoader.load': 1,
    'swift.model.models.llama.LlamaLoader.get_config': 1,
    'swift.template.register.get_template': 1,
    'swift.pipelines.train.sft.SwiftSft.run': 1,
    'swift.trainers.trainer_factory.TrainerFactory.get_trainer_cls': 2,
    'swift.trainers.seq2seq_trainer.Seq2SeqTrainer.compute_loss': 3,
    'swift.template.base.Template.compute_sft_loss': 3,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--runtime', choices=('torch', 'jittor'), default='torch')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.model = args.model.resolve()
    if not (args.root / 'compat/tests/torch/_ecosystem_runner.py').is_file():
        parser.error('--root must contain the maintained ecosystem runner')
    targets = [args.model, *([args.output.resolve()] if args.output else [])]
    if any(path == args.root or args.root in path.parents for path in targets):
        parser.error('model/output artifacts must be outside the checkout')
    if args.action == 'prepare' and (args.runtime != 'torch' or args.device != 'cpu'):
        parser.error('initial checkpoint preparation requires genuine Torch CPU')
    if args.action == 'run' and not all((args.upstream, args.wheel, args.output)):
        parser.error('run requires --upstream, --wheel and --output')
    return args


def prepare(torch, args):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    args.model.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                         max_position_embeddings=128, attention_dropout=0.,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         tie_word_embeddings=False)
    model = LlamaForCausalLM(config)
    assert all(value.device.type == 'cpu' for value in model.parameters())
    model.save_pretrained(args.model)
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
    vocab.update({f'word{i}': i + 4 for i in range(60)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                       pad_token='<pad>', bos_token='<s>', eos_token='</s>')
    tokenizer.save_pretrained(args.model)
    print('ECOSYSTEM_RESULT ' + json.dumps({'case': CASE, 'action': 'prepared',
          'model_artifacts': {path.name: digest(path)
              for path in sorted(args.model.iterdir()) if path.is_file()}}))


def run(torch, runner, stack, args):
    import numpy as np
    from component_measure import installed_distribution
    from registry_inventory import inventory
    from swift.arguments import SftArguments
    from swift.pipelines.train.sft import SwiftSft
    from swift.trainers import TrainerFactory
    from transformers import TrainerCallback
    import swift

    tf32 = runner._configure_tf32(torch, args.device)
    baseline = subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip()
    def clean():
        if subprocess.check_output(['git', '-C', str(args.root), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Swift measurements require a clean source checkout')
        if subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip() != baseline:
            raise ValueError('Source HEAD changed during measurement')
    clean()
    source = inventory('ms-swift', args.upstream)
    binding = installed_distribution(source, swift)
    # Bind the actual wheel payload, including non-Python template/dataset data.
    with zipfile.ZipFile(args.wheel) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel members')
        payload = {name[len('swift/'):]: hashlib.sha256(wheel.read(name)).hexdigest()
                   for name in names if name.startswith('swift/') and not name.endswith('/')}
        headers = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(headers) != 1:
            raise ValueError('Expected one actual Swift wheel distribution')
        header = BytesParser().parsebytes(wheel.read(headers[0]))
        if (header['Name'] or '').replace('_', '-').lower() != 'ms-swift' or header['Version'] != binding['version']:
            raise ValueError('Swift wheel distribution name/version mismatch')
    if payload != binding['installed_files_sha256']:
        raise ValueError('Actual installed Swift payload differs from the bound wheel')
    for relative, expected in payload.items():
        original = args.upstream / 'swift' / relative
        if not original.is_file() or digest(original) != expected:
            raise ValueError('Swift wheel payload differs from the clean upstream tree: ' + relative)
    binding['wheel_sha256'] = digest(args.wheel)
    binding['payload_source_sha256'] = {'swift/' + name: value for name, value in payload.items()}
    before = None
    scope = nullcontext()
    if args.runtime == 'jittor':
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        stack.enter_context(jt.runtime.scope(backend_fallback='error'))
        before = jt.core.backend_fallback_count()
        scope = forbid_backend_fallbacks()
        if Path(jt.__file__).resolve() != args.root / 'python/jittor/__init__.py':
            raise ValueError('Actual Jittor does not belong to the requested source tree')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.with_name(args.output.stem + '.work')
    work.mkdir(exist_ok=False)
    rng = np.random.RandomState(args.seed)
    records = [{'messages': [{'role': 'user', 'content': ' '.join(f'word{i}' for i in rng.randint(0, 60, 3))},
                             {'role': 'assistant', 'content': ' '.join(f'word{i}' for i in rng.randint(0, 60, 3))}]}
               for _ in range(3)]
    dataset = work / 'train.jsonl'
    dataset.write_text(''.join(json.dumps(row) + '\n' for row in records))
    arrays, observed, identity, observed_inputs = {}, {}, {}, {}

    def snapshot(value, name):
        if value.device.type != args.device:
            raise AssertionError('Actual tensor device mismatch: ' + name)
        if args.runtime == 'jittor':
            value.sync()
            if value.location() != ('device' if args.device == 'cuda' else 'cpu'):
                raise AssertionError('Physical tensor residency mismatch: ' + name)
            if jt.core.dispatch_context([value])[0] != args.device:
                raise AssertionError('Native dispatch device mismatch: ' + name)
        result = np.array(value.detach().cpu().numpy(), copy=True)
        if not np.isfinite(result).all():
            raise AssertionError('Non-finite tensor: ' + name)
        observed[name] = args.device
        return result

    class Observer(TrainerCallback):
        def __init__(self, model):
            self.model, self.outputs, self.embeddings = model, [], []
            self.steps, self.updates = 0, 0
            self.parameters = dict(model.named_parameters())
            self.buffers = dict(model.named_buffers())
            if len(self.parameters) != 12 or not all(p.requires_grad for p in self.parameters.values()):
                raise AssertionError('Changed fully trainable tiny Llama parameter scope')
            if not all(p.dtype == torch.float32 for p in self.parameters.values()):
                raise AssertionError('The declared Trainer model is not actually FP32')
            for name, value in {**self.parameters, **self.buffers}.items():
                arrays['initial::' + name] = snapshot(value, 'initial::' + name)
            def observe_embedding(module, inputs, output):
                output.retain_grad()
                self.embeddings.append(output)
            def observe_model(module, inputs, kwargs, output):
                if inputs:
                    raise AssertionError('Unexpected positional model inputs in the real Trainer')
                prefix = 'step/%d/' % self.steps
                for name, value in kwargs.items():
                    if not torch.is_tensor(value):
                        continue
                    key = prefix + 'observed_input::' + name
                    data = snapshot(value, key)
                    if data.dtype.kind not in 'biu' or value.requires_grad:
                        raise AssertionError('Unexpected floating/grad-requiring model input: ' + name)
                    observed_inputs[key] = {'dtype': str(data.dtype), 'shape': list(data.shape),
                        'values': data.tolist(), 'sha256': hashlib.sha256(data.tobytes()).hexdigest()}
                self.outputs.append(output)
            self.hooks = [model.get_input_embeddings().register_forward_hook(observe_embedding),
                          model.register_forward_hook(observe_model, with_kwargs=True)]

        def on_pre_optimizer_step(self, training_args, state, control, **kwargs):
            step = int(state.global_step)
            if step != self.steps or len(self.outputs) != 1 or len(self.embeddings) != 1:
                raise AssertionError('Unexpected real Trainer forward/step order')
            prefix = 'step/%d/' % step
            output, embedding = self.outputs[0], self.embeddings[0]
            for name, value in [('output', output.logits), ('loss', output.loss),
                                ('observed_grad::embedding_output', embedding.grad)]:
                if value is None:
                    raise AssertionError('Missing requested observation: ' + name)
                arrays[prefix + name] = snapshot(value, prefix + name)
            for name, value in self.parameters.items():
                if value.grad is None:
                    raise AssertionError('Missing requested parameter gradient: ' + name)
                arrays[prefix + 'grad::' + name] = snapshot(value.grad, prefix + 'grad::' + name)
            self.steps += 1

        def on_step_end(self, training_args, state, control, **kwargs):
            step = int(state.global_step) - 1
            if step != self.updates or self.steps != self.updates + 1:
                raise AssertionError('Unexpected optimizer update order')
            for name, value in self.parameters.items():
                key = 'step/%d/parameter::%s' % (step, name)
                arrays[key] = snapshot(value, key)
            self.outputs.clear()
            self.embeddings.clear()
            self.updates += 1

    class ObservedSft(SwiftSft):
        def train(self, trainer):
            qualified = type(trainer).__module__ + '.' + type(trainer).__name__
            selected = TrainerFactory.TRAINER_MAPPING[self.args.task_type]
            if type(trainer) is not TrainerFactory.get_trainer_cls(self.args):
                raise AssertionError('Actual Trainer is not the selected registered Trainer')
            identity.update(model_family=trainer.model.model_meta.model_type,
                            task=trainer.model.model_info.task_type, trainer=self.args.task_type,
                            trainer_class=qualified, trainer_registry_target=selected,
                            model_class=type(trainer.model).__module__ + '.' + type(trainer.model).__name__,
                            loader_class=trainer.model.model_meta.loader.__module__ + '.' + trainer.model.model_meta.loader.__name__,
                            template_type=trainer.template.template_meta.template_type)
            observer = Observer(trainer.model)
            trainer.add_callback(observer)
            try:
                result = super().train(trainer)
            finally:
                for handle in observer.hooks:
                    handle.remove()
            if trainer.state.global_step != 3 or observer.steps != 3 or observer.updates != 3:
                raise AssertionError('Real Swift Trainer did not complete three updates')
            identity.update(trainable_parameters=sorted(observer.parameters), frozen_parameters=[],
                            buffer_names=sorted(observer.buffers), optimizer_updates=observer.updates)
            return result

    calls = {name: 0 for name in UPSTREAM_CALLS}

    def observe_upstream(frame, event, _argument):
        if event == 'call':
            name = frame.f_globals.get('__name__', '') + '.' + frame.f_code.co_qualname
            if name in calls:
                calls[name] += 1

    with scope:
        sft = SftArguments(model=str(args.model), model_type='llama', template='llama',
            dataset=[str(dataset)], split_dataset_ratio=0., use_hf=True,
            output_dir=str(work/'trainer'), add_version=False, tuner_type='full',
            torch_dtype='float32', use_cpu=args.device == 'cpu', bf16=False, fp16=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=3,
            max_length=128, learning_rate=.001, weight_decay=0., optim='sgd',
            lr_scheduler_type='constant', warmup_steps=0, max_grad_norm=0.,
            gradient_checkpointing=False, dataset_shuffle=False, train_dataloader_shuffle=False,
            dataloader_num_workers=0, dataloader_pin_memory=False, dataset_num_proc=1,
            logging_steps=1, disable_tqdm=True, report_to=[], save_strategy='no',
            eval_strategy='no', check_model=False, use_logits_to_keep=False,
            tf32=False, seed=0, data_seed=0)
        previous_profile = sys.getprofile()
        sys.setprofile(observe_upstream)
        try:
            ObservedSft(sft).main()
        finally:
            sys.setprofile(previous_profile)
        runner._synchronize(torch, args.runtime, args.device)
    if calls != UPSTREAM_CALLS:
        raise AssertionError('Actual upstream loader/template/Trainer calls changed: ' + json.dumps(calls))
    refs = [{'repo': 'ms-swift', 'axis': axis, 'member': identity[axis]}
            for axis in ('model_family', 'trainer', 'task')]
    if not all(any(a['axis'] == ref['axis'] and ref['member'] in a['members'] for a in source['axes']) for ref in refs):
        raise AssertionError('Actual loader/task/Trainer identity is absent from the pinned registry')
    fallback_count = jt.core.backend_fallback_count() - before if before is not None else 0
    if fallback_count:
        raise AssertionError('Backend fallback observed')
    after = installed_distribution(source, swift)
    if any(after.get(key) != value for key, value in binding.items()
           if key not in ('wheel_sha256', 'payload_source_sha256')):
        raise ValueError('Actual installed Swift changed during training')
    if inventory('ms-swift', args.upstream) != source:
        raise ValueError('Pinned upstream source changed during training')
    clean()
    with args.output.open('xb') as stream:
        np.savez(stream, **arrays)
    binary = getattr(getattr(torch, '_C', None), '__file__', None) if args.runtime == 'torch' else None
    if args.runtime == 'torch' and (not binary or not Path(binary).is_file()):
        raise AssertionError('Oracle must load a genuine binary PyTorch extension')
    actual_precision = {'matmul': bool(torch.backends.cuda.matmul.allow_tf32) if args.device == 'cuda' else False,
        'cudnn': bool(torch.backends.cudnn.allow_tf32) if args.device == 'cuda' else False,
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark) if args.device == 'cuda' else False,
        'matmul_precision': torch.get_float32_matmul_precision()}
    if actual_precision != tf32:
        raise AssertionError('Precision policy changed during the real Trainer execution')
    metadata = {'case': CASE, 'runtime': args.runtime, 'device': args.device,
        'jittor_commit': baseline, 'upstream': source['upstream'], 'registry_refs': refs,
        'downstream_distribution': binding, 'steps': 3, 'seed': args.seed, 'learning_rate': .001,
        'mode': 'train', 'trainable_parameters': identity['trainable_parameters'],
        'frozen_parameters': [], 'input_gradient_names': [],
        'observed_intermediate_gradients': ['embedding_output'],
        'integer_inputs': sorted({key.split('::')[1] for key in observed_inputs}),
        'observed_inputs': observed_inputs,
        'training_protocol': {'owner': 'swift.pipelines.train.sft.SwiftSft.main',
            'trainer_owner': 'swift.trainers.Seq2SeqTrainer.train', 'optimizer': 'torch.optim.SGD',
            'scheduler': 'constant', 'gradient_clipping': False, 'identity': identity,
            'upstream_calls': calls},
        'model_directory': str(args.model),
        'model_artifacts': {path.name: digest(path) for path in sorted(args.model.iterdir()) if path.is_file()},
        'dataset': {'path': str(dataset), 'sha256': digest(dataset)},
        'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE') == '1',
        'dependencies': runner._dependency_report(('swift', 'transformers', 'peft', 'accelerate')),
        'tf32': tf32, 'runtime_conditions': runner._runtime_conditions(torch, tf32),
        'backend': runner._backend_report(args.runtime), 'fallback_count': fallback_count,
        'fallback_policy': 'error' if args.runtime == 'jittor' else None,
        'observed_tensor_devices': observed,
        'interpreter': {'executable': sys.executable, 'pid': os.getpid(),
            'is_shim': hasattr(torch, '_torch_compat_install_context'), 'torch_version': torch.__version__,
            'binary_extension': binary, 'binary_sha256': digest(binary) if binary else None},
        'measurement_command': [sys.executable, *sys.argv],
        'measurement_source_sha256': {str(path.relative_to(args.root)): digest(path) for path in
            (Path(__file__).resolve(), args.root/'compat/tests/torch/_ecosystem_runner.py',
             args.root/'agent/skills/torch-compat-repo-runbook/scripts/component_measure.py',
             args.root/'agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py')}}
    if args.runtime == 'jittor':
        metadata['native_runtime_flags'] = {name: int(getattr(jt.flags, name)) for name in
            ('use_cuda', 'auto_graph_replay', 'auto_graph_replay_retain_bytes')}
    write_json(args.output.with_suffix('.json'), metadata)
    print('ECOSYSTEM_RESULT ' + json.dumps(metadata))


def main(argv=None):
    args = options(argv)
    if args.device == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', use_cuda='0', JT_USE_CUDA='0',
                          nvcc_path='', JT_BUILD_NVCC_PATH='')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise ValueError('Both hub offline flags must be explicitly enabled')
    sys.path.insert(0, str(args.root/'compat/tests/torch'))
    import _ecosystem_runner as runner
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        runner._configure_tf32(torch, args.device)
        torch.set_num_threads(1)
        if args.action == 'prepare':
            prepare(torch, args)
        else:
            run(torch, runner, stack, args)


if __name__ == '__main__':
    main()
