"""
train_compare.py -- ICRNNA vs MCSBA on RadioML 2016.10a, trained under one common protocol.

Both models are trained on the SAME fixed split with the SAME seeds, then evaluated on the same test frames.
Summary statistics (mean +/- std, Welch test, SNR bands) are written to results/ at the end of the run.

Models
  * ICRNNA (baseline, El-Haryqy et al., Results in Engineering 26 (2025) 104783):
      Conv1D 64 -> Conv1D 128 -> BiLSTM x2 -> additive attention -> Dense 128 -> Dense 64 -> softmax
      (794,188 parameters for 11 classes)
  * MCSBA (proposed):
      multi-scale Conv1D (k = 3/5/7) + concatenation -> two Conv1D blocks with SE -> BiLSTM x2
      -> improved additive attention -> Dense 256 -> Dense 128 -> softmax
      (942,620 parameters for 11 classes)

Protocol (identical for both models)
  * split seed 2016, stratified by modulation class, 70/15/15
  * standardisation statistics from the TRAINING split only (one mean/std per I/Q channel)
  * batch size 64, up to 100 epochs, Adam (lr 1e-3, clipnorm 1.0)
  * EarlyStopping (val_loss, patience 13), ReduceLROnPlateau (val_loss, factor 0.3, patience 4, min_lr 1e-7)
  * the checkpoint with the highest val_accuracy is used for the test evaluation
  * initialisation seeds: 42, 123, 2024 (the models are different, so pairing by seed is nominal)

Usage
  1. Download RML2016.10a_dict.pkl (DeepSig) and set FILE_PATH below.
  2. python train_compare.py
Output: results/ (per-seed JSON, weights, summaries, comparison figures)
"""

import csv, json, os, pickle, time
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Concatenate, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, Input, Lambda, Layer, LayerNormalization, LSTM,
    MaxPooling1D, Multiply
)
from tensorflow.keras.regularizers import l2

# =============================================================
# CONFIG
# =============================================================
FILE_PATH   = "/content/drive/MyDrive/RML2016.10a/RML2016.10a_dict.pkl"
OUT_DIR     = "results"
SPLIT_SEED  = 2016
EPS         = 1e-7

SEEDS       = [42, 123, 2024]        # 3 runs per model, SAME seeds for pairing
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
CLIPNORM    = 1.0
ES_PATIENCE = 13
LR_FACTOR   = 0.3
LR_PATIENCE = 4
MIN_LR      = 1e-7

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================
# DATA
# =============================================================
def load_and_prepare():
    with open(FILE_PATH, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    mods     = sorted({k[0] for k in d})      # deterministic order
    snr_list = sorted({k[1] for k in d})

    X, y, s = [], [], []
    for mi, m in enumerate(mods):
        for snr in snr_list:
            if (m, snr) in d:
                arr = np.asarray(d[(m, snr)], dtype=np.float32)
                X.append(arr)
                y.append(np.full(len(arr), mi, dtype=np.int64))
                s.append(np.full(len(arr), snr, dtype=np.int64))
    X = np.concatenate(X); y = np.concatenate(y); s = np.concatenate(s)
    assert X.shape[1:] == (2, 128), X.shape

    # --- 70/15/15 stratified split (same for BOTH models) ---
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30,
                               random_state=SPLIT_SEED, stratify=y)
    va, te  = train_test_split(tmp, test_size=0.50,
                               random_state=SPLIT_SEED, stratify=y[tmp])

    # --- Normalisation from TRAIN only, per I/Q channel ---
    mu = X[tr].mean(axis=(0, 2), keepdims=True)
    sd = X[tr].std(axis=(0, 2), keepdims=True)
    Xtr = (X[tr] - mu) / (sd + EPS)
    Xva = (X[va] - mu) / (sd + EPS)
    Xte = (X[te] - mu) / (sd + EPS)

    return Xtr, y[tr], Xva, y[va], Xte, y[te], s[te], mods, snr_list, \
           mu.ravel().tolist(), sd.ravel().tolist()


# =============================================================
# SHARED ATTENTION MODULE
# =============================================================
class ImprovedAdditiveAttention(Layer):
    def __init__(self, units, use_scale=True, dropout_rate=0.1, **kw):
        super().__init__(**kw)
        self.units, self.use_scale, self.dropout_rate = units, use_scale, dropout_rate
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.V  = Dense(1, use_bias=False)
        if use_scale:
            self.scale = self.add_weight(name="scale", shape=(),
                                         initializer="ones", trainable=True)
        self.dropout    = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, query, value, training=False):
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(value)))
        if self.use_scale:
            score = score * self.scale
        w   = tf.nn.softmax(score, axis=1)
        w   = self.dropout(w, training=training)
        ctx = tf.reduce_sum(w * value, axis=1)
        return self.layer_norm(ctx)


# =============================================================
# ICRNNA baseline
# =============================================================
def build_icrnna(num_classes, input_shape=(2, 128)):
    inp = Input(shape=input_shape)
    x = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(inp)

    x = Conv1D(64, 3, activation="relu", padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, activation="relu", padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(1e-4)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    lstm_out = Bidirectional(LSTM(128, return_sequences=True,
                                  kernel_regularizer=l2(1e-4)))(x)
    x = BatchNormalization()(lstm_out)
    x = Dropout(0.3)(x)

    x = ImprovedAdditiveAttention(128, use_scale=True, dropout_rate=0.1)(x, lstm_out)

    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR, clipnorm=CLIPNORM),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# =============================================================
# MCSBA (proposed model)
# =============================================================
class SqueezeExcitation(Layer):
    def __init__(self, ratio=8, **kw):
        super().__init__(**kw)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.fc1 = Dense(c // self.ratio, activation="relu")
        self.fc2 = Dense(c, activation="sigmoid")

    def call(self, x):
        se = self.fc2(self.fc1(self.squeeze(x)))
        return Multiply()([x, tf.reshape(se, [-1, 1, tf.shape(x)[-1]])])


def build_mcsba(num_classes, input_shape=(2, 128)):
    inp = Input(shape=input_shape)
    x = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(inp)

    b1 = BatchNormalization()(Conv1D(48, 3, activation="relu", padding="same",
                                     kernel_regularizer=l2(1e-4))(x))
    b2 = BatchNormalization()(Conv1D(48, 5, activation="relu", padding="same",
                                     kernel_regularizer=l2(1e-4))(x))
    b3 = BatchNormalization()(Conv1D(48, 7, activation="relu", padding="same",
                                     kernel_regularizer=l2(1e-4))(x))
    x = Dropout(0.3)(Concatenate()([b1, b2, b3]))

    x = Conv1D(128, 3, activation="relu", padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = SqueezeExcitation(8)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, activation="relu", padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = SqueezeExcitation(8)(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.3)(LayerNormalization(epsilon=1e-6)(x))

    lstm_out = Bidirectional(LSTM(128, return_sequences=True,
                                  kernel_regularizer=l2(1e-4)))(x)
    x = Dropout(0.3)(LayerNormalization(epsilon=1e-6)(lstm_out))

    x = ImprovedAdditiveAttention(128, use_scale=True, dropout_rate=0.1)(x, lstm_out)

    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(BatchNormalization()(x))
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(BatchNormalization()(x))

    out = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR, clipnorm=CLIPNORM),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


BUILDERS = {
    "ICRNNA": build_icrnna,
    "MCSBA":  build_mcsba,
}


# =============================================================
# EVALUATION
# =============================================================
def evaluate(pred, y, s, snr_list, mods):
    res = {
        "acc": float((pred == y).mean()),
        "f1":  float(f1_score(y, pred, average="weighted")),
        "precision": float(precision_score(y, pred, average="weighted",
                                           zero_division=0)),
        "per_snr_acc": {}, "per_snr_f1": {}, "per_class_acc": {},
    }
    for snr in snr_list:
        m = s == snr
        if m.any():
            res["per_snr_acc"][str(snr)] = float((pred[m] == y[m]).mean())
            res["per_snr_f1"][str(snr)]  = float(f1_score(y[m], pred[m],
                                                           average="weighted"))
    for i, name in enumerate(mods):
        m = y == i
        res["per_class_acc"][name] = float((pred[m] == y[m]).mean())
    return res


def run_seed(model_name, seed, Xtr, ytr, Xva, yva, Xte, yte, ste,
             mods, snr_list, num_classes):
    tag  = f"{model_name}_10a_seed{seed}"
    path = os.path.join(OUT_DIR, tag + ".json")
    if os.path.exists(path):
        print(f"[{tag}] already done -> loading")
        with open(path) as f:
            return json.load(f)

    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = BUILDERS[model_name](num_classes)
    print(f"params = {model.count_params():,}")

    ckpt = os.path.join(OUT_DIR, tag + ".weights.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                      mode="min", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR,
                          patience=LR_PATIENCE, min_lr=MIN_LR),
        ModelCheckpoint(ckpt, monitor="val_accuracy", mode="max",
                        save_best_only=True, save_weights_only=True, verbose=0),
    ]

    t0 = time.time()
    hist = model.fit(Xtr, tf.keras.utils.to_categorical(ytr, num_classes),
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_data=(Xva, tf.keras.utils.to_categorical(yva, num_classes)),
                     callbacks=callbacks, verbose=2)
    elapsed = time.time() - t0

    model.load_weights(ckpt)
    pred = model.predict(Xte, batch_size=1024, verbose=0).argmax(1)
    res  = evaluate(pred, yte, ste, snr_list, mods)
    res.update(seed=seed,
               epochs_run=len(hist.history["loss"]),
               best_val_acc=float(max(hist.history["val_accuracy"])),
               params=int(model.count_params()),
               train_seconds=float(elapsed))

    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{tag}] test acc = {res['acc']*100:.2f}%  F1 = {res['f1']*100:.2f}%  "
          f"epochs = {res['epochs_run']}  time = {elapsed/60:.1f} min")
    return res


# =============================================================
# AGGREGATION
# =============================================================
def mean_std_ci(vals, conf=0.95):
    v = np.asarray(vals, dtype=float); n = len(v)
    m  = v.mean()
    sd = v.std(ddof=1) if n > 1 else 0.0
    h  = stats.t.ppf((1+conf)/2., n-1) * sd / np.sqrt(n) if n > 1 else 0.0
    return m, sd, h


def summarise(runs, model_name, snr_list, mods):
    summ = {"model": model_name, "dataset": "10a",
            "seeds": [r["seed"] for r in runs], "n_runs": len(runs),
            "protocol": {"batch_size": BATCH_SIZE, "epochs": EPOCHS,
                         "lr": LR, "es_patience": ES_PATIENCE,
                         "lr_factor": LR_FACTOR, "lr_patience": LR_PATIENCE,
                         "min_lr": MIN_LR, "split_seed": SPLIT_SEED},
            "runs": runs}

    print(f"\n=== {model_name} on RadioML 2016.10a  (n={len(runs)} seeds) ===")
    for key in ("acc", "f1", "precision"):
        vals = [r[key]*100 for r in runs]
        m, sd, h = mean_std_ci(vals)
        summ[key] = {"mean": m, "std": sd, "ci95_halfwidth": h, "values": vals}
        print(f"{key:10s}: {m:.2f} ± {sd:.2f}  (95% CI ± {h:.2f})  "
              f"runs={[round(v,2) for v in vals]}")

    rows = []
    for snr in snr_list:
        a = [r["per_snr_acc"][str(snr)]*100 for r in runs if str(snr) in r["per_snr_acc"]]
        if a:
            rows.append([snr, np.mean(a), np.std(a, ddof=1) if len(a)>1 else 0])
    with open(os.path.join(OUT_DIR, f"{model_name}_10a_per_snr.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh); w.writerow(["snr_db", "acc_mean", "acc_std"]); w.writerows(rows)

    with open(os.path.join(OUT_DIR, f"{model_name}_10a_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)

    print(f"\n{model_name} per-class accuracy (mean ± std):")
    for mod in mods:
        vals = [r["per_class_acc"][mod]*100 for r in runs]
        m, sd, h = mean_std_ci(vals)
        print(f"  {mod:10s}: {m:.2f} ± {sd:.2f}")

    return summ


# =============================================================
# COMPARISON + FIGURES
# =============================================================
def compare_and_plot(icrnna_runs, mcsba_runs, mods, snr_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({r["seed"] for r in icrnna_runs} & {r["seed"] for r in mcsba_runs})
    a = {r["seed"]: r for r in icrnna_runs}
    b = {r["seed"]: r for r in mcsba_runs}

    print(f"\n{'='*70}\nPAIRED COMPARISON  ICRNNA  vs  MCSBA"
          f"\nseeds = {seeds}\n{'='*70}")
    for key in ("acc", "f1"):
        d = np.array([(b[s][key] - a[s][key])*100 for s in seeds])   # MCSBA - ICRNNA
        m, sd, h = mean_std_ci(d)
        t, p = stats.ttest_rel([b[s][key] for s in seeds],
                               [a[s][key] for s in seeds])
        print(f"{key.upper()}: Δ(MCSBA − ICRNNA) = {m:+.2f} pp  95% CI ± {h:.2f}  "
              f"p = {p:.4f}   per-seed = {np.round(d, 2).tolist()}")
        if p < 0.05:
            print("  -> statistically significant at α=0.05")
        else:
            print("  -> NOT significant at α=0.05")

    # --- Figure 1: overall accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ic_m = np.mean([a[s]["acc"] for s in seeds]) * 100
    ic_s = np.std ([a[s]["acc"] for s in seeds], ddof=1) * 100
    mc_m = np.mean([b[s]["acc"] for s in seeds]) * 100
    mc_s = np.std ([b[s]["acc"] for s in seeds], ddof=1) * 100
    ax.bar([0, 1], [ic_m, mc_m], yerr=[ic_s, mc_s], capsize=8,
           color=["tab:blue", "tab:orange"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ICRNNA", "MCSBA"])
    ax.set_ylabel("Overall test accuracy (%)")
    ax.set_title(f"ICRNNA vs MCSBA - RadioML 2016.10a\n"
                 f"(mean ± std over {len(seeds)} seeds)")
    for i, (v, sd) in enumerate([(ic_m, ic_s), (mc_m, mc_s)]):
        ax.text(i, v + sd + 0.5, f"{v:.2f}±{sd:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_bar_overall.png"), dpi=200)
    plt.close()

    # --- Figure 2: accuracy vs SNR ---
    snrs = sorted({int(s) for s in snr_list})
    ic_m = np.array([np.mean([a[s]["per_snr_acc"][str(x)] for s in seeds]) for x in snrs]) * 100
    ic_s = np.array([np.std ([a[s]["per_snr_acc"][str(x)] for s in seeds], ddof=1) for x in snrs]) * 100
    mc_m = np.array([np.mean([b[s]["per_snr_acc"][str(x)] for s in seeds]) for x in snrs]) * 100
    mc_s = np.array([np.std ([b[s]["per_snr_acc"][str(x)] for s in seeds], ddof=1) for x in snrs]) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(snrs, ic_m, marker="o", label="ICRNNA", color="tab:blue")
    plt.fill_between(snrs, np.clip(ic_m - ic_s, 0, 100), np.clip(ic_m + ic_s, 0, 100),
                     alpha=0.20, color="tab:blue")
    plt.plot(snrs, mc_m, marker="s", label="MCSBA", color="tab:orange")
    plt.fill_between(snrs, np.clip(mc_m - mc_s, 0, 100), np.clip(mc_m + mc_s, 0, 100),
                     alpha=0.20, color="tab:orange")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy (%)")
    plt.title(f"ICRNNA vs MCSBA on 2016.10a (mean ± std, {len(seeds)} seeds)")
    plt.grid(alpha=0.3); plt.legend(); plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_acc_vs_snr.png"), dpi=200)
    plt.close()

    # --- Figure 3: per-class accuracy ---
    x  = np.arange(len(mods)); w = 0.38
    ic   = [np.mean([a[s]["per_class_acc"][m] for s in seeds])*100 for m in mods]
    ic_e = [np.std ([a[s]["per_class_acc"][m] for s in seeds], ddof=1)*100 for m in mods]
    mc   = [np.mean([b[s]["per_class_acc"][m] for s in seeds])*100 for m in mods]
    mc_e = [np.std ([b[s]["per_class_acc"][m] for s in seeds], ddof=1)*100 for m in mods]

    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, ic, w, yerr=ic_e, capsize=3, label="ICRNNA", color="tab:blue")
    plt.bar(x + w/2, mc, w, yerr=mc_e, capsize=3, label="MCSBA",  color="tab:orange")
    plt.xticks(x, mods, rotation=45, ha="right")
    plt.ylabel("Per-class accuracy (%)")
    plt.title("Per-class accuracy - ICRNNA vs MCSBA (mean ± std)")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_per_class.png"), dpi=200)
    plt.close()

    print(f"\nFigures saved in {OUT_DIR}/")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    (Xtr, ytr, Xva, yva, Xte, yte, ste,
     mods, snr_list, mu, sd) = load_and_prepare()

    print(f"\nmods ({len(mods)}): {mods}")
    print(f"SNR levels: {snr_list}")
    print(f"train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")
    print(f"train-only mu = {mu}, sd = {sd}")
    print(f"models = ICRNNA | MCSBA")
    print(f"seeds  = {SEEDS}\n")

    icrnna_runs, mcsba_runs = [], []

    for seed in SEEDS:
        icrnna_runs.append(run_seed("ICRNNA", seed, Xtr, ytr, Xva, yva, Xte, yte,
                                    ste, mods, snr_list, len(mods)))
    for seed in SEEDS:
        mcsba_runs.append(run_seed("MCSBA", seed, Xtr, ytr, Xva, yva, Xte, yte,
                                   ste, mods, snr_list, len(mods)))

    summarise(icrnna_runs, "ICRNNA", snr_list, mods)
    summarise(mcsba_runs,  "MCSBA",  snr_list, mods)

    compare_and_plot(icrnna_runs, mcsba_runs, mods, snr_list)

    print("\nAll results saved to results/")
    print("  ICRNNA_10a_seed{42,123,2024}.json")
    print("  MCSBA_10a_seed{42,123,2024}.json")
    print("  ICRNNA_10a_summary.json / MCSBA_10a_summary.json")
    print("  comparison_bar_overall.png")
    print("  comparison_acc_vs_snr.png")
    print("  comparison_per_class.png")