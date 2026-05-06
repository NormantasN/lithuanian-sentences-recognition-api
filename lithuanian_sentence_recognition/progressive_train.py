# -*- coding: utf-8 -*-
import os
import random
import json
from datetime import datetime

import tensorflow as tf
try:
    [tf.config.experimental.set_memory_growth(gpu, True)
     for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from mltu.configs import BaseModelConfigs
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.annotations.images import CVImage
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from model import train_model

CONFIGS_PATH  = "Models/LT_Sentence_Recognition/202603222204/configs.yaml"
REAL_CSV      = "Datasets/Real_Handwriting/annotations.csv"
SYNTH_CSV     = "Datasets/Synthetic_Sentences/annotations.csv"
RESULTS_FILE  = "progressive_fromscratch_results.json"

N_RUNS        = 10
SYNTH_PER_RUN = 2500   # 0 = 1:0, 2500 = 1:1, 5000 = 1:2
HOLDOUT_SIZE  = 250

HEIGHT, WIDTH = 199, 2262
BATCH_SIZE    = 32
LEARNING_RATE = 0.0005
MAX_EPOCHS    = 100

pretrained_configs = BaseModelConfigs.load(CONFIGS_PATH)
VOCAB   = pretrained_configs.vocab
max_len = pretrained_configs.max_text_length
print(f"Vocab ({len(VOCAB)} simboliu): {VOCAB}")

def load_csv(path):
    data = []
    if not os.path.exists(path):
        print(f"Nerastas: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            img_path, label = parts
            if os.path.exists(img_path):
                data.append([img_path, label])
    return data


real_data  = load_csv(REAL_CSV)
synth_data = load_csv(SYNTH_CSV)

print(f"Realiu duomenu:    {len(real_data)}")
print(f"Sintetiniu (pool): {len(synth_data)}")

random.seed(42)
real_data_shuffled = real_data.copy()
random.shuffle(real_data_shuffled)
holdout_data = real_data_shuffled[:HOLDOUT_SIZE]
train_pool   = real_data_shuffled[HOLDOUT_SIZE:]

print(f"Holdout (neliečiamas): {len(holdout_data)}")
print(f"Treniravimo pool:      {len(train_pool)}")

def make_provider(dataset):
    return DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=BATCH_SIZE,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(WIDTH, HEIGHT, keep_aspect_ratio=True),
            LabelIndexer(VOCAB),
            LabelPadding(max_word_length=max_len, padding_value=len(VOCAB)),
        ]
    )


def evaluate_holdout(model):
    provider = make_provider(holdout_data)
    results  = model.evaluate(provider, verbose=0)
    return float(results[1]), float(results[2])  # CER, WER

all_results        = []
current_model_path = None

for run_idx in range(N_RUNS):
    print(f"\n{'='*60}")
    print(f"RUN {run_idx + 1}/{N_RUNS}")
    print(f"{'='*60}")

    train_shuffled = train_pool.copy()
    random.shuffle(train_shuffled)
    synth_sample = random.sample(synth_data, min(SYNTH_PER_RUN, len(synth_data)))

    dataset = train_shuffled + synth_sample
    random.shuffle(dataset)

    print(f"Duomenys: {len(train_shuffled)} realiu + {len(synth_sample)} sint. = {len(dataset)} is viso")

    run_model_path = os.path.join(
        "Models/LT_Progressive_Scratch",
        f"run_{run_idx+1:02d}_{datetime.strftime(datetime.now(), '%Y%m%d%H%M')}"
    )
    os.makedirs(run_model_path, exist_ok=True)

    if current_model_path is None:
        print("Treniruojamas nuo nulio...")
        model = train_model(input_dim=(HEIGHT, WIDTH, 3), output_dim=len(VOCAB))
        lr    = LEARNING_RATE
    else:
        print(f"Uzkraunamas: {current_model_path}")
        model = tf.keras.models.load_model(
            current_model_path, compile=False, safe_mode=False
        )
        lr = LEARNING_RATE * 0.1

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=CTCloss(),
        metrics=[
            CERMetric(vocabulary=VOCAB),
            WERMetric(vocabulary=VOCAB),
        ],
        run_eagerly=False,
    )

    data_provider        = make_provider(dataset)
    train_dp, val_dp     = data_provider.split(split=0.9)
    train_dp.augmentors  = [
        RandomBrightness(),
        RandomErodeDilate(),
        RandomSharpen(),
    ]

    print(f"LR: {lr} | Train: {len(train_dp)} batchu | Val: {len(val_dp)} batchu")

    saved_model = f"{run_model_path}/model.h5"

    callbacks = [
        EarlyStopping(monitor="val_CER", patience=15, verbose=1, mode="min"),
        ModelCheckpoint(saved_model, monitor="val_CER", verbose=1,
                        save_best_only=True, mode="min"),
        TrainLogger(run_model_path),
        ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10,
                          patience=5, verbose=1, mode="min"),
        Model2onnx(saved_model),
    ]

    history = model.fit(
        train_dp,
        validation_data=val_dp,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )

    train_dp.to_csv(os.path.join(run_model_path, "train.csv"))
    val_dp.to_csv(os.path.join(run_model_path,   "val.csv"))

    best_val_cer = min(history.history.get("val_CER", [float("inf")]))
    best_epoch   = history.history.get("val_CER", []).index(best_val_cer) + 1

    best_model = tf.keras.models.load_model(saved_model, compile=False, safe_mode=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=CTCloss(),
        metrics=[CERMetric(vocabulary=VOCAB), WERMetric(vocabulary=VOCAB)],
    )
    holdout_cer, holdout_wer = evaluate_holdout(best_model)
    print(f"Holdout CER: {holdout_cer:.4f} | Holdout WER: {holdout_wer:.4f}")

    run_result = {
        "run":           run_idx + 1,
        "saved_to":      saved_model,
        "best_val_CER":  float(best_val_cer),
        "holdout_CER":   holdout_cer,
        "holdout_WER":   holdout_wer,
        "best_epoch":    best_epoch,
        "total_epochs":  len(history.history.get("val_CER", [])),
        "real_count":    len(train_shuffled),
        "synth_count":   len(synth_sample),
        "learning_rate": lr,
    }
    all_results.append(run_result)
    current_model_path = saved_model

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"PROGRESYVUS TRENIRAVIMAS BAIGTAS ({N_RUNS} runs)")
print(f"{'='*60}")
print(f"{'Run':>4} | {'val_CER':>8} | {'holdout_CER':>11} | {'holdout_WER':>11} | {'LR':>8}")
print(f"{'-'*55}")
for r in all_results:
    print(f"  {r['run']:>2} | {r['best_val_CER']:>8.4f} | "
          f"{r['holdout_CER']:>11.4f} | {r['holdout_WER']:>11.4f} | "
          f"{r['learning_rate']:>8.6f}")

hc = [r["holdout_CER"] for r in all_results]
print(f"\nHoldout CER: {hc[0]:.4f} -> {hc[-1]:.4f}  (pagerėjimas: {hc[0]-hc[-1]:.4f})")
print(f"Galutinis modelis: {current_model_path}")
print(f"Rezultatai: {RESULTS_FILE}")