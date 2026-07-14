# Command-line guide

The installed command, module entry point, and root scripts are equivalent:

```bash
famous-cnns list
python -m famous_cnns list
python scripts/train.py --help
```

## Classification data

Use an ImageFolder-style layout:

```text
dataset/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

```bash
famous-cnns train \
  --model resnet50 \
  --data-dir dataset \
  --num-classes 2 \
  --epochs 20 \
  --batch-size 32 \
  --optimizer adamw \
  --lr 0.0003 \
  --output outputs/resnet50.pt
```

## Segmentation data

U-Net expects matching filename stems in `images` and `masks`:

```text
dataset/
├── train/
│   ├── images/sample_001.jpg
│   └── masks/sample_001.png
└── val/
    ├── images/sample_002.jpg
    └── masks/sample_002.png
```

```bash
famous-cnns train --model unet --data-dir dataset --num-classes 1 --image-size 256
```

Binary masks use zero as background and any positive value as foreground. Multiclass masks must contain integer class IDs.

## Inference

```bash
famous-cnns infer \
  --checkpoint outputs/resnet50.pt \
  --input samples/ \
  --top-k 3 \
  --output outputs/predictions.json
```

For segmentation, inference saves a tensor file containing input paths and predicted masks.

## Architecture scripts

Each architecture has `scripts/train.py` and `scripts/infer.py`. The train wrapper selects its model automatically:

```bash
python lenet/scripts/train.py --num-classes 10 --data-dir dataset
python mobilenet/scripts/train.py --v2 --num-classes 196 --data-dir dataset
python resnet/scripts/train.py --resnet101 --num-classes 37 --data-dir dataset
python efficient-net/scripts/train.py --num-classes 101 --model-kwargs '{"variant":"b3"}' --data-dir dataset
```

Inference reconstructs the architecture from checkpoint metadata, so the same interface is used in every folder.

## Smoke tests

No dataset is downloaded when `--smoke-test` is supplied:

```bash
famous-cnns train --model lenet5 --num-classes 2 --epochs 1 --batch-size 2 --smoke-test
famous-cnns infer --checkpoint outputs/lenet5.pt --smoke-test
```
