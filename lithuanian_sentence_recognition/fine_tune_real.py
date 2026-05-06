import os
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard, ReduceLROnPlateau)
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from mltu.configs import BaseModelConfigs
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.annotations.images import CVImage

PRETRAINED_MODEL = "Models/LT_Sentence_Recognition/202512071555/model.h5"
REAL_CSV         = "Datasets/Real_Handwriting/annotations.csv"
SYNTH_CSV        = "Datasets/Synthetic_Sentences/annotations.csv"
MAX_SYNTH        = 7500


class FineTuneConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/LT_Sentence_Recognition",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M") + "_finetune"
        )
        self.vocab           = " aąbcčdeęėfghiįyjklmnoprsštuųūvzž.,!?-"
        self.height          = 96
        self.width           = 1408
        self.max_text_length = 0
        self.batch_size      = 8
        self.learning_rate   = 0.00005
        self.train_epochs    = 100
        self.train_workers   = 20


def load_csv(path, limit=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) != 2:
                continue
            img_path, label = parts
            if os.path.exists(img_path):
                data.append([img_path, label])
    if limit:
        random.shuffle(data)
        data = data[:limit]
    return data


real_data  = load_csv(REAL_CSV)
synth_data = load_csv(SYNTH_CSV, limit=MAX_SYNTH)
dataset    = real_data + synth_data
random.shuffle(dataset)

print(f"Realiu duomenu:     {len(real_data)}")
print(f"Sintetiniu duomenu: {len(synth_data)}")
print(f"Is viso:            {len(dataset)}")

vocab = set()
max_len = 0
for _, label in dataset:
    vocab.update(list(label))
    max_len = max(max_len, len(label))

configs = FineTuneConfigs()
configs.vocab            = "".join(sorted(vocab))
configs.max_text_length  = max_len
configs.save()

print(f"Zodynas:    {configs.vocab}")
print(f"Max ilgis:  {max_len}")

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length,
                     padding_value=len(configs.vocab)),
    ],
)

train_data_provider, val_data_provider = data_provider.split(split=0.9)
print(f"Treniravimo: {len(train_data_provider)} batchu")
print(f"Validavimo:  {len(val_data_provider)} batchu")

train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
]

print(f"\nIkeliamas modelis: {PRETRAINED_MODEL}")
model = tf.keras.models.load_model(PRETRAINED_MODEL, compile=False, safe_mode=False)

print("Uzsaldomi konvoliuciniai sluoksniai")
for layer in model.layers:
    freeze = any(x in layer.name for x in ['residual', 'conv', 'lambda', 'reshape'])
    layer.trainable = not freeze

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab),
    ],
)

os.makedirs(configs.model_path, exist_ok=True)

callbacks_phase1 = [
    EarlyStopping(monitor="val_CER", patience=10, verbose=1, mode="min"),
    ModelCheckpoint(f"{configs.model_path}/model.h5",
                    monitor="val_CER", verbose=1,
                    save_best_only=True, mode="min"),
    TrainLogger(configs.model_path),
    ReduceLROnPlateau(monitor="val_CER", factor=0.7,
                      patience=4, verbose=1, mode="min"),
    TensorBoard(f"{configs.model_path}/logs_phase1", update_freq=1),
]

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=50,
    callbacks=callbacks_phase1,
)

print("\n Treniruojami visi sluoksniai")
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=configs.learning_rate / 5),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab),
    ],
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_CER", patience=15, verbose=1, mode="min"),
    ModelCheckpoint(f"{configs.model_path}/model.h5",
                    monitor="val_CER", verbose=1,
                    save_best_only=True, mode="min"),
    TrainLogger(configs.model_path),
    ReduceLROnPlateau(monitor="val_CER", factor=0.8,
                      patience=5, verbose=1, mode="min"),
    TensorBoard(f"{configs.model_path}/logs_phase2", update_freq=1),
    Model2onnx(f"{configs.model_path}/model.h5"),
]

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=100,
    callbacks=callbacks_phase2,
)

train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

print(f"\nFine-tuning baigtas.")
print(f"Modelis: {configs.model_path}/model.h5")
print(f"ONNX:    {configs.model_path}/model.onnx")