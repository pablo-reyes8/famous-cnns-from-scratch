# Repository architecture

The repository has two intentionally separate layers.

```text
architecture folders              unified library
lenet/                             famous_cnns/factory.py
alexnet/                           famous_cnns/orchestrator.py
vgg/                ───────────▶   famous_cnns/cli.py
incpetion/                         famous_cnns/data.py
resnet/                            famous_cnns/visualization.py
u-net/
mobilenet/
efficient-net/
```

Architecture folders preserve the from-scratch, paper-oriented implementations. `famous_cnns` provides stable construction, channel adaptation, training, evaluation, checkpoint, visualization, and CLI behavior.

The registry is the integration boundary. A model entry declares its import path, class, task, default channels, recommended image size, aliases, and optional custom builder. Shared code must not import architecture-specific training notebooks or datasets.

Checkpoints contain state dictionaries and reconstruction metadata. Loading uses PyTorch's restricted `weights_only` mode; do not replace this with unrestricted pickle loading for untrusted files.
