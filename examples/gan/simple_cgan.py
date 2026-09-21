"""Train a small conditional GAN and serve it, doing no work at import time.

The task and the network are the same as ``examples/notebooks/conditional_gan.md``:
an 8x8 image whose bright column is the requested class. That keeps the demo
runnable offline -- nothing is downloaded, and the historical pretrained
checkpoints are gone -- while still showing the whole path from training a
conditional generator to wiring it into an app.
"""

from __future__ import print_function

import argparse
import sys
from pathlib import Path


#: Conditional classes. Class ``k`` means "the k-th column is the bright one".
CLASS_COUNT = 8
IMAGE_SIZE = 8
LATENT_DIM = 16
BATCH_SIZE = 64
TRAIN_STEPS = 400
LEARNING_RATE = 2e-3
DIGITS = "01234567"
#: An 8x8 tile is unreadable at native size; scale it up for the saved/rendered image.
DISPLAY_SCALE = 12


def _parse_digits(value, class_count=CLASS_COUNT):
    """Reject anything the generator has no condition for."""
    allowed = DIGITS[:class_count]
    if not value or any(character not in allowed for character in value):
        raise ValueError(
            "digits must contain only characters 0 through %d" % (class_count - 1))
    return value


def _sample_real(np, batch, labels=None):
    """The model's world: a +1 column at ``labels``, -1 elsewhere, plus noise.

    The +/-1 range matches the generator's final ``Tanh``, the same as the
    tutorial does it.
    """
    if labels is None:
        labels = np.random.randint(0, CLASS_COUNT, size=batch)
    images = np.full((batch, 1, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype="float32")
    images[np.arange(batch), 0, :, labels] = 1.0
    images += np.random.normal(0, 0.05, images.shape).astype("float32")
    return images, labels


def _build_models(jt, nn):
    """Generator and discriminator, both conditioned on the class."""

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.embed = nn.Embedding(CLASS_COUNT, LATENT_DIM)
            self.net = nn.Sequential(
                nn.Linear(LATENT_DIM * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, IMAGE_SIZE * IMAGE_SIZE),
                nn.Tanh(),
            )

        def execute(self, noise, labels):
            condition = self.embed(labels)
            hidden = self.net(jt.concat((noise, condition), dim=1))
            return hidden.reshape((hidden.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE))

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.embed = nn.Embedding(CLASS_COUNT, IMAGE_SIZE * IMAGE_SIZE)
            self.net = nn.Sequential(
                nn.Linear(IMAGE_SIZE * IMAGE_SIZE * 2, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
            )

        def execute(self, images, labels):
            flat = images.reshape((images.shape[0], -1))
            condition = self.embed(labels)
            return self.net(jt.concat((flat, condition), dim=1))

    return Generator(), Discriminator()


def _train(jt, nn, np, generator, discriminator, steps, seed=0, progress=None):
    """The CGAN training loop: one discriminator step, then one generator step."""
    jt.set_global_seed(seed)
    np.random.seed(seed)

    bce = nn.BCEWithLogitsLoss()
    g_optim = jt.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    d_optim = jt.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    report_every = max(1, steps // 8)

    for step in range(steps):
        images_np, labels_np = _sample_real(np, BATCH_SIZE)
        real_images = jt.array(images_np)
        real_labels = jt.array(labels_np).int32()

        noise = jt.randn((BATCH_SIZE, LATENT_DIM))
        fake_images = generator(noise, real_labels)
        d_real = discriminator(real_images, real_labels)
        d_fake = discriminator(fake_images.stop_grad(), real_labels)
        d_loss = bce(d_real, jt.ones_like(d_real)) + bce(d_fake, jt.zeros_like(d_fake))
        d_optim.step(d_loss)

        noise = jt.randn((BATCH_SIZE, LATENT_DIM))
        fake_images = generator(noise, real_labels)
        d_fake = discriminator(fake_images, real_labels)
        g_loss = bce(d_fake, jt.ones_like(d_fake))
        g_optim.step(g_loss)

        if progress is not None and step % report_every == 0:
            progress(step, steps, d_loss.item(), g_loss.item())

    return generator


def _render(jt, np, generator, digits):
    """Render one generated tile per requested class, joined into a strip."""
    labels = np.array([int(character) for character in digits])
    noise = jt.randn((len(labels), LATENT_DIM))
    generated = generator(noise, jt.array(labels).int32()).numpy()
    generated = (generated + 1) / 2                        # [-1, 1] -> [0, 1]
    strip = np.concatenate([generated[index, 0] for index in range(len(labels))], axis=1)
    strip = (strip * 255).clip(0, 255).astype("uint8")
    strip = np.repeat(np.repeat(strip, DISPLAY_SCALE, axis=0), DISPLAY_SCALE, axis=1)
    return np.stack([strip] * 3, axis=-1)                  # grayscale -> RGB


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", default=DIGITS)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.steps < 1:
        parser.error("--steps must be positive")
    try:
        _parse_digits(args.digits)
    except ValueError as error:
        parser.error(str(error))

    import numpy as np
    import jittor as jt
    from jittor import nn

    generator, discriminator = _build_models(jt, nn)

    def progress(step, steps, d_loss, g_loss):
        sys.stderr.write(
            "step %4d/%d  D损失=%.3f  G损失=%.3f\n" % (step, steps, d_loss, g_loss))
        sys.stderr.flush()

    sys.stderr.write("training on CPU, about a minute at the default --steps\n")
    _train(jt, nn, np, generator, discriminator, args.steps,
           seed=args.seed, progress=progress)
    generator.eval()

    if args.output is not None:
        from PIL import Image

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_render(jt, np, generator, args.digits)).save(str(args.output))
        print("wrote {}".format(args.output))

    if args.no_server:
        return 0

    try:
        from PIL import Image
        import pywebio as pw
    except ImportError as error:
        raise SystemExit(
            "The web demo requires the packages in requirements/examples.txt"
        ) from error

    def web_server():
        pw.pin.put_input("digits", label="Digits to generate (0-%d):" % (CLASS_COUNT - 1))

        def generate(_value):
            try:
                digits = _parse_digits(pw.pin.pin.digits)
            except ValueError as error:
                pw.output.put_error(str(error))
                return
            pw.output.put_image(Image.fromarray(_render(jt, np, generator, digits)))

        pw.output.put_buttons(["Generate"], onclick=generate)

    pw.start_server(web_server, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
