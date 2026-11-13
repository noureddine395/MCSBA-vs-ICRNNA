# MCSBA vs ICRNNA: controlled comparison on RadioML 2016.10a

Code for the controlled three-seed comparison of ICRNNA and MCSBA in the paper
*Enhanced Modulation Recognition via Multi-Scale Convolutional Features with Channel Attention*.
`train_compare.py` contains both models, the data preparation, the training setup and the per-seed evaluation.

## Run

1. Download `RML2016.10a_dict.pkl` from https://www.deepsig.ai/datasets.
2. Set `FILE_PATH` (and optionally `OUT_DIR`) at the top of `train_compare.py`.
3. `pip install -r requirements.txt`, then `python train_compare.py`.

Outputs in `results/`: one JSON per model and seed, weights, mean/std summaries, per-SNR CSV files and comparison figures.

## Protocol

Split seed 2016 (stratified by class, 70/15/15); standardisation from the training split only (per I/Q channel);
Adam 1e-3, clipnorm 1.0, batch 64, up to 100 epochs; EarlyStopping (val_loss, patience 13);
ReduceLROnPlateau (factor 0.3, patience 4, min 1e-7); test on the checkpoint of highest validation accuracy;
seeds 42, 123, 2024 for both models. Parameters: ICRNNA 794,188; MCSBA 942,620.

## Expected result (test accuracy, mean ± std)

| Model | Accuracy (%) | Per seed (42, 123, 2024) |
|---|---|---|
| ICRNNA | 62.43 ± 0.60 | 62.72, 61.73, 62.83 |
| MCSBA | 63.87 ± 0.15 | 63.76, 63.81, 64.04 |

Difference +1.44 points (Welch 95% CI [0.05, 2.83], p = 0.047). Small deviations are possible with other TensorFlow or GPU versions.

## Other datasets and the paper's single-run results

For RadioML 2016.10b or RML22, change `FILE_PATH`, `SPLIT_SEED`, `SEEDS`, `BATCH_SIZE`, `ES_PATIENCE`, `LR_FACTOR`, `LR_PATIENCE`.
Settings used in the paper: 10b single runs: batch 256, LR factor 0.3, LR patience 2, early-stop patience 15;
RML22: split seed 2022, batch 512, LR factor 0.2, LR patience 4, early-stop patience 13, seeds 123, 42, 2024, 7, 88.

The single-run (10a, 10b) and RML22 results use the original protocol, which differs from this script in two places:

1. In `load_and_prepare`, standardise with whole-dataset statistics per time sample, computed before splitting:
   `mu = X.mean(axis=(0, 1), keepdims=True)`, `sd = X.std(axis=(0, 1), keepdims=True)`.
2. In `run_seed`, delete `model.load_weights(ckpt)` so the test set uses the early-stopping weights.

The single runs on 10a and 10b were not seeded or split reproducibly, so they cannot be reproduced bit for bit.

## Data and licence

Datasets are not redistributed; use the DeepSig licence for RadioML. RML22: Sathyanarayanan et al.,
IEEE Trans. Wireless Commun. 22(11), 2023, https://doi.org/10.1109/TWC.2023.3254490. Code licence: MIT.
