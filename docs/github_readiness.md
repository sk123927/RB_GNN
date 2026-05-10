# GitHub Readiness Notes

This cleaned repository is organized for external review:

- No hard-coded absolute paths.
- No backup scripts or experiment scratch files.
- One canonical data reader for all supervised models.
- Small enough bundled data to clone quickly.
- Clear configs for main model, mask model, and baselines.
- Tests cover the most important file-format contract.

Suggested before publishing:

1. Create a fresh virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python scripts/inspect_dataset.py --dataset trigger/er1500`.
4. Run `python -m unittest discover -s tests`.
5. Train one tiny smoke run:

```bash
python scripts/train.py --dataset demo/tiny --model rbgnn --epochs 2 --batch-size 1 --hidden-dim 16 --num-layers 2
```

