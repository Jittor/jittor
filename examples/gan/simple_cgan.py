"""Serve a pretrained conditional GAN without doing work at import time."""

from __future__ import print_function

import argparse
from pathlib import Path


DEFAULT_GENERATOR_WEIGHTS = "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/generator_last.pkl"
DEFAULT_DISCRIMINATOR_WEIGHTS = (
    "https://cg.cs.tsinghua.edu.cn/jittor/assets/build/discriminator_last.pkl"
)


def _build_models(jt, nn, np, latent_dim, class_count, image_size, channels):
    image_shape = (channels, image_size, image_size)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.label_embedding = nn.Embedding(class_count, class_count)

            def block(input_features, output_features, normalize=True):
                layers = [nn.Linear(input_features, output_features)]
                if normalize:
                    layers.append(nn.BatchNorm1d(output_features, 0.8))
                layers.append(nn.LeakyReLU(0.2))
                return layers

            self.model = nn.Sequential(
                *block(latent_dim + class_count, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(image_shape))),
                nn.Tanh(),
            )

        def execute(self, noise, labels):
            generated = self.model(jt.concat((self.label_embedding(labels), noise), dim=1))
            return generated.view((generated.shape[0], *image_shape))

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.label_embedding = nn.Embedding(class_count, class_count)
            self.model = nn.Sequential(
                nn.Linear(class_count + int(np.prod(image_shape)), 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 1),
            )

        def execute(self, image, labels):
            flattened = image.view((image.shape[0], -1))
            inputs = jt.concat((flattened, self.label_embedding(labels)), dim=1)
            return self.model(inputs)

    return Generator(), Discriminator()


def _generate_image(jt, np, generator, digits, latent_dim):
    if not digits or any(character not in "0123456789" for character in digits):
        raise ValueError("digits must contain only characters 0 through 9")
    count = len(digits)
    noise = jt.array(np.random.normal(0, 1, (count, latent_dim))).float32().stop_grad()
    labels = jt.array(np.array([int(character) for character in digits])).int32().stop_grad()
    images = generator(noise, labels)
    images = images.transpose((1, 2, 0, 3)).reshape(images.shape[2], -1)
    images = images[:, :, None].broadcast(images.shape + (3,))
    images = (images - images.min()) / (images.max() - images.min()) * 255
    return images.uint8().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", default="201962517")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--generator-weights", default=DEFAULT_GENERATOR_WEIGHTS)
    parser.add_argument("--discriminator-weights", default=DEFAULT_DISCRIMINATOR_WEIGHTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    import numpy as np
    import jittor as jt
    from jittor import nn

    latent_dim = 100
    generator, discriminator = _build_models(
        jt,
        nn,
        np,
        latent_dim=latent_dim,
        class_count=10,
        image_size=32,
        channels=1,
    )
    generator.eval()
    discriminator.eval()
    generator.load(args.generator_weights)
    discriminator.load(args.discriminator_weights)

    if args.output is not None:
        from PIL import Image

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_generate_image(jt, np, generator, args.digits, latent_dim)).save(
            str(args.output)
        )
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
        pw.pin.put_input("number", label="Digits to generate:")

        def generate(_value):
            try:
                image = _generate_image(jt, np, generator, pw.pin.pin.number, latent_dim)
            except ValueError as error:
                pw.output.put_error(str(error))
                return
            pw.output.put_image(Image.fromarray(image))

        pw.output.put_buttons(["Generate"], onclick=generate)

    pw.start_server(web_server, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
