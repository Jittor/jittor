"""A common image and sample pipeline for cross-framework timing audits.

Keep JPEG decoding and log-uniform RandomResizedCrop on CPU. A sample's
crop depends on its pass and index, so worker assignment cannot change it.
Workers transport RGB bytes. The benchmark scales them on the accelerator.
"""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image


def crop_box(height, width, rng, scale=(.5, 1.), ratio=(3./4., 4./3.)):
    area = height * width
    for _ in range(10):
        target = area * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w, h = int(round(math.sqrt(target*aspect))), int(round(math.sqrt(target/aspect)))
        if 0 < w <= width and 0 < h <= height:
            return rng.randint(0, height-h), rng.randint(0, width-w), h, w
    aspect = width / height
    if aspect < min(ratio):
        w, h = width, int(round(width/min(ratio)))
    elif aspect > max(ratio):
        h, w = height, int(round(height*max(ratio)))
    else:
        h, w = height, width
    return (height-h)//2, (width-w)//2, h, w


def image_array(filename, sample, pass_index=0, seed=42, training=True):
    with Image.open(filename) as source:
        image = source.convert('RGB')
    if training:
        rng = random.Random(f'{seed}:{pass_index}:{sample}')
        top, left, height, width = crop_box(image.height, image.width, rng)
        image = image.crop((left, top, left+width, top+height))
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))


class SharedPets:
    def __init__(self, files, labels, training=True, seed=42):
        self.files, self.labels = list(files), list(labels)
        self.training, self.seed = training, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        import torch
        pass_index, index = key if isinstance(key, tuple) else (0, key)
        pixels = image_array(self.files[index], index, pass_index, self.seed, self.training)
        return {'image': torch.from_numpy(pixels), 'label': self.labels[index]}


class SharedBatches:
    drop_last = False
    def __init__(self, samples, batch_size, cycles, seed=42):
        self.samples, self.batch_size, self.cycles, self.seed = samples, batch_size, cycles, seed
        self.pass_index = 0

    def __len__(self):
        return self.cycles * math.ceil(self.samples/self.batch_size)

    def __iter__(self):
        for _ in range(self.cycles):
            current = self.pass_index
            self.pass_index += 1
            order = np.random.RandomState(self.seed+current).permutation(self.samples).tolist()
            for start in range(0, self.samples, self.batch_size):
                yield [(current, index) for index in order[start:start+self.batch_size]]

    def set_epoch(self, epoch):
        self.pass_index = int(epoch)*self.cycles


def plan_hash(training, evaluation, cycles):
    plan = dict(training=[(Path(p).name, y) for p, y in zip(training.files, training.labels)],
                evaluation=[(Path(p).name, y) for p, y in zip(evaluation.files, evaluation.labels)],
                seed=training.seed, cycles=cycles, size=[224,224], scale=[.5,1.],
                ratio=[3./4.,4./3.], aspect_distribution='log-uniform', interpolation='PIL bilinear')
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
