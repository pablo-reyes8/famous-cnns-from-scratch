# Unified scripts

These are filesystem-friendly alternatives to the installed `famous-cnns` command.

```bash
python scripts/train.py --model lenet5 --num-classes 10 --data-dir data/mnist
python scripts/infer.py --checkpoint outputs/lenet5.pt --input sample.png
python -m famous_cnns list
```

Run either command with `--help` for the complete interface. Architecture folders contain thin wrappers that select their model automatically.
