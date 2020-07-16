import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from fake_useragent import UserAgent
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
import jittor as jt

try:
    from tqdm.auto import tqdm  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):  # type: ignore

            def __init__(self, total=None, disable=False,
                         unit=None, unit_scale=None, unit_divisor=None):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
                sys.stderr.flush()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write('\n')

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
MASTER_BRANCH = 'master'
DEFAULT_CACHE_DIR = '~/.cache'
_hub_dir = None
useragent = UserAgent()

def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner, repo_name, branch)

def _get_jittor_home():
    return os.path.expanduser(os.path.join(DEFAULT_CACHE_DIR, 'jittor'))

def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch

def _get_cache_or_reload(github, reload):
    # Setup hub_dir to save downloaded files
    hub_dir = get_dir()
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = branch.replace('/', '_')
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(hub_dir, '_'.join([repo_owner, repo_name, normalized_br]))

    use_cache = (not reload) and os.path.exists(repo_dir)

    if use_cache:
        sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, normalized_br + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)
        sys.stderr.write('Downloading: \"{}\" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=False)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir

def get_dir():
    r"""Get the Jittor Hub cache directory used for storing downloaded models & weights.
        If :func:`set_dir` is not called, default path is ``_get_jittor_home()/hub``.
    """
    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_jittor_home(), 'hub')


def set_dir(d):
    r"""Optionally set the Jittor Hub directory used to save downloaded models & weights.
    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = d


def load_model(github, model, *args, **kwargs):
    r"""Load a model from a github repo, with pretrained weights.
    Args:
        github (string): a string with format "repo_owner/repo_name:branch". Branch is optional. The default branch is `master` if not specified.
            Example: 'Jittor/gan-jittor:jittorhub'
        model (string): a string of entrypoint name.
            Example: 'models.cgan.models:Generator'
        *args (optional): the corresponding args for callable `model`.
        reload (bool, optional): whether to force a fresh download of github repo unconditionally.
            Default is `False`.
        **kwargs (optional): the corresponding kwargs for callable `model`.
    Returns:
        a single model with corresponding pretrained weights.
    Example:
        model = jt.hub.load('Jittor/gan-jittor:jittorhub', 'models.cgan.models:Generator', load_pretrained_weight=True)
    """
    reload = kwargs.get('reload', False)
    kwargs.pop('reload', None)
    load_pretrained_weight = kwargs.get('load_pretrained_weight', False)
    kwargs.pop('load_pretrained_weight', None)

    repo_dir = _get_cache_or_reload(github, reload)
    sys.path.insert(0, repo_dir)
    model_path, model_name = model.split(':')

    # import model class
    exec(f"from {model_path} import {model_name}")
    # instance model
    loc = locals()
    exec(f"instance_model = {model_name}(*args, **kwargs)")
    instance_model = loc['instance_model']

    if load_pretrained_weight:
        if 'jittorhub.py' not in os.listdir(repo_dir):
            raise RuntimeError(f"No support pretrained weight for {model}.")
        else:
            from jittorhub import pretrained_weights_url
            if model in pretrained_weights_url.keys():
                weight_url = pretrained_weights_url[model]
                weights = load_weight_from_url(weight_url)
                instance_model.load_parameters(weights)
            else:
                raise RuntimeError(f"No support pretrained weight for {model}.")
    
    sys.path.remove(repo_dir)
    return instance_model

def load_module(github, model, reload=False):
    r"""Load module from a github repo, with pretrained weights.
    Args:
        github (string): a string with format "repo_owner/repo_name:branch". Branch is optional. The default branch is `master` if not specified.
            Example: 'Jittor/gan-jittor:jittorhub'
        model (string): a string of entrypoint name.
            Example: 'models.cgan.models:Generator'
        reload (bool, optional): whether to force a fresh download of github repo unconditionally.
            Default is `False`.
    Example:
        Generator = jt.hub.load_module('Jittor/gan-jittor:jittorhub', 'models.cgan.models')
        print(Generator)
    
    You will get the following results:
        <class 'models.cgan.models.Generator'>
    """
    repo_dir = _get_cache_or_reload(github, reload)
    sys.path.insert(0, repo_dir)
    
    loc = locals()
    model_path, model_name = model.split(':')
    exec(f"from {model_path} import {model_name}")
    return loc[model_name]

def load_env(github, reload=False):
    r"""Load environment from a github repo, with pretrained weights.
    Args:
        github (string): a string with format "repo_owner/repo_name:branch". Branch is optional. The default branch is `master` if not specified.
            Example: 'Jittor/gan-jittor:jittorhub'
        reload (bool, optional): whether to force a fresh download of github repo unconditionally.
            Default is `False`.
    Example:
        jt.hub.load_env('Jittor/gan-jittor:jittorhub')
        from models.cgan.models import *
        print(Generator)
        print(Discriminator)
    
    You will get the following results:
        <class 'models.cgan.models.Generator'>
        <class 'models.cgan.models.Discriminator'>
    """
    repo_dir = _get_cache_or_reload(github, reload)
    sys.path.insert(0, repo_dir)

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    Example:
        jt.hub.download_url_to_file('https://cg.cs.tsinghua.edu.cn/jittor/assets/build/generator_last.pkl', '/tmp/temporary_file')
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": useragent.random})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def load_weight_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    r"""Loads the Jittor weights at the given URL.
    If downloaded file is a zip file, it will be automatically
    decompressed.
    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`get_dir`.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.
    Example:
        weights = jt.hub.load_weight_from_url('https://cg.cs.tsinghua.edu.cn/jittor/assets/build/generator_last.pkl')
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1) if check_hash else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)
    return jt.load(cached_file)