"""Model selection and deterministic numerical probes for training benchmarks."""
import hashlib


def file_digest(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numerical_model_options(config):
    """Disable model-config dropout before constructing a numerical reference.

    T5 attention, for example, reads a float rather than an nn.Dropout module.
    Only existing probability fields become overrides accepted by from_pretrained.
    """
    options = {}
    for name, value in config.to_dict().items():
        probability = ('dropout' in name or name.endswith('_pdrop')
                       or name in ('encoder_layerdrop', 'decoder_layerdrop', 'drop_path_rate'))
        if probability and (value is None or isinstance(value, (int, float))):
            options[name] = 0.0
    return options


def disable_stochastic_layers(model, torch):
    """Numerical probes also disable timm DropPath and module-held probabilities."""
    changed = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            changed.append(name + '.p')
        for attribute in ('drop_prob', 'dropout', 'attention_dropout'):
            value = getattr(module, attribute, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(module, attribute, 0.0)
                changed.append(name + '.' + attribute)
    return changed


def use_math_attention_reference(torch):
    """Select the independent FP32 math oracle only for numerical probes."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def install_compiled_features(model, torch):
    """Compile frozen features while preserving each model's calling contract."""
    original = model.forward_features
    runners = {}

    def features(inputs, *args, **kwargs):
        key = (bool(model.training), tuple(inputs.shape), str(inputs.dtype))
        if key not in runners:
            runners[key] = torch.compile(original, mode='reduce-overhead', dynamic=False)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            return runners[key](inputs, *args, **kwargs)

    model.forward_features = features
    return runners


def install_tokenizer_selection(auto_tokenizer, model, revision=None):
    """Use the selected model's tokenizer inside the unmodified official loader.

    Decoder classifiers need a padding token. Reusing their EOS token keeps the
    vocabulary and pretrained embedding size unchanged on both frameworks.
    """
    original = auto_tokenizer.from_pretrained
    observed = []

    def load(_official_name, *args, **kwargs):
        if revision is not None:
            kwargs['revision'] = revision
        tokenizer = original(model, *args, **kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError('The selected tokenizer has neither padding nor EOS: ' + model)
            tokenizer.pad_token = tokenizer.eos_token
        observed.append(dict(model=model, revision=revision,
                             tokenizer_class=type(tokenizer).__name__,
                             pad_token_id=tokenizer.pad_token_id,
                             model_input_names=list(tokenizer.model_input_names)))
        return tokenizer

    auto_tokenizer.from_pretrained = load
    return observed
