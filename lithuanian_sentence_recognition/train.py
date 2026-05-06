import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from mltu.annotations.images import CVImage
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from mltu.configs import BaseModelConfigs
from mltu.preprocessors import ImageReader
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from model import train_model


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/LT_Sentence_Recognition",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )
        self.vocab = " ,.?aąbcčdeęėfghiįyjklmnoprsštuųūvzžAĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ"
        self.height = 199
        self.width = 2262
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.train_epochs = 200
        self.train_workers = 8


REAL_CSV = "Datasets/Real_Handwriting/annotations.csv"
SYNTH_CSV = "Datasets/Synthetic_Sentences/annotations.csv"

dataset, vocab, max_len = [], set(), 0

for csv_path in [REAL_CSV, SYNTH_CSV]:
    if not os.path.exists(csv_path):
        print(f"Praleistas (nerastas): {csv_path}")
        continue

    found, missing = 0, 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            image_path, label = parts
            if not os.path.exists(image_path):
                missing += 1
                if missing <= 100:
                    print(f"  Nerastas: {image_path}")
                continue
            dataset.append([image_path, label])
            vocab.update(list(label))
            max_len = max(max_len, len(label))
            found += 1

    print(f"  {csv_path}: rasta={found}, nerasta={missing}")

print(f"Iš viso pavyzdžių: {len(dataset)}")
print(f"Unikalių simbolių: {len(vocab)}")
print(f"Max sakinio ilgis: {max_len}")

configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ]
)

train_data_provider, val_data_provider = data_provider.split(split=0.9)

print(f"Treniravimo batches: {len(train_data_provider)}")
print(f"Validavimo batches:  {len(val_data_provider)}")

train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
]

model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab),
    ],
    run_eagerly=False,
)

model.summary(line_length=110)

os.makedirs(configs.model_path, exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_CER", patience=15, verbose=1, mode="min"),
    ModelCheckpoint(
        f"{configs.model_path}/model.h5",
        monitor="val_CER", verbose=1, save_best_only=True, mode="min",
    ),
    TrainLogger(configs.model_path),
    ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="min"),
    Model2onnx(f"{configs.model_path}/model.h5"),
]

print("\n Pradedamas mokymas...")
history = model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=callbacks,
)

train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

print(f"\n Mokymas baigtas!")
print(f"Modelis: {configs.model_path}/model.h5")
print(f"ONNX:    {configs.model_path}/model.onnx")
