import os
from datetime import datetime

import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tqdm import tqdm

from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from mltu.configs import BaseModelConfigs
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.annotations.images import CVImage

from model import train_model


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/LT_Sentence_Recognition",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )
        self.vocab = " aąbcčdeęėfghiįyjklmnoprsštuųūvzž.,!?-"
        self.height = 96
        self.width = 1408
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 200
        self.train_workers = 20


dataset_path = "Datasets/LT_Sentences/annotations.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Duomenų rinkinys nerastas: {dataset_path}\n"
        f"Pirma paleiskite generate_lithuanian_sentences.py"
    )

dataset, vocab, max_len = [], set(), 0

print("Kraunami duomenys...")
with open(dataset_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in tqdm(lines):
    line = line.strip()
    if not line:
        continue

    parts = line.split(',', 1)
    if len(parts) != 2:
        continue

    image_path, label = parts

    if not os.path.exists(image_path):
        print(f"Failas nerastas: {image_path}")
        continue

    dataset.append([image_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

print(f"\nIš viso pavyzdžių: {len(dataset)}")
print(f"Unikalių simbolių: {len(vocab)}")
print(f"Maksimalus sakinio ilgis: {max_len}")
print(f"Rasti simboliai: {''.join(sorted(vocab))}")
print("\n🔍 Tikriname ar visi failai egzistuoja...")
missing_files = []
for img_path, label in dataset[:100]:
    if not os.path.exists(img_path):
        missing_files.append(img_path)
        if len(missing_files) <= 5:
            print(f"Nerastas: {img_path}")

if missing_files:
    print(f"\n⚠ Nerasta {len(missing_files)}/100 failų!")
else:
    print(f"✓ Visi 100 failų rasti!")

print("\nBandome nuskaityti kelis vaizdus...")
for i in range(min(100, len(dataset))):
    img_path, label = dataset[i]
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"cv2.imread grąžino None: {img_path}")
        else:
            print(f"✓ Pavyzdys {i}: {img.shape}, label: '{label}'")
    except Exception as e:
        print(f"Pavyzdys {i} klaida: {e}")
configs = ModelConfigs()

configs.vocab = "".join(sorted(vocab))
print(f"\nŽODYNO ANALIZĖ:")
print(f"Žodynas ({len(configs.vocab)} simbolių): {configs.vocab}")
print(f"Ar yra tarpas: {' ' in configs.vocab}")
print(f"Ar yra didžiosios: {any(c.isupper() for c in configs.vocab)}")

print("\nTikriname ar visi label simboliai yra žodyne...")
labels_missing_chars = 0
for img_path, label in dataset[:100]:
    missing = [c for c in label if c not in configs.vocab]
    if missing:
        labels_missing_chars += 1
        if labels_missing_chars <= 3:
            print(f"Label '{label}' turi simbolių ne iš žodyno: {set(missing)}")

print(f"Iš 100 pirmų: {labels_missing_chars} turi simbolių ne iš žodyno")
configs.max_text_length = max_len
configs.save()

print(f"\nModelio konfigūracija išsaugota: {configs.model_path}")

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ],
)
print(f"\nDataProvider DEBUG:")
print(f"Dataset dydis DataProvider viduje: {len(data_provider._dataset)}")
print(f"DataProvider.__len__(): {len(data_provider)}")
print(f"Batch size: {configs.batch_size}")
print(f"Tikėtinas batch'ų skaičius: {len(dataset) // configs.batch_size}")
print(f"Realus batch'ų skaičius: {len(data_provider)}")

# Pabandome gauti pirmą batch
print("\n🔍 Bandome gauti pirmą batch...")
try:
    first_batch = data_provider[0]
    print(f"✓ Pirmas batch gautas: {len(first_batch)} elementai")
    if len(first_batch) == 2:
        images, labels = first_batch
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
except Exception as e:
    print(f"Klaida gaunant batch: {e}")
    import traceback

    traceback.print_exc()
# Padalijame į mokymo ir validavimo rinkinius
train_data_provider, val_data_provider = data_provider.split(split=0.9)

print(f"\nMokymo pavyzdžių: {len(train_data_provider)}")
print(f"Validavimo pavyzdžių: {len(val_data_provider)}")

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
        WERMetric(vocabulary=configs.vocab)
    ],
    run_eagerly=False
)

print("\nModelio architektūra:")
model.summary(line_length=110)

earlystopper = EarlyStopping(monitor="val_CER", patience=15, verbose=1, mode="min")
checkpoint = ModelCheckpoint(
    f"{configs.model_path}/model.h5",
    monitor="val_CER",
    verbose=1,
    save_best_only=True,
    mode="min"
)
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_CER",
    factor=0.8,
    min_delta=1e-10,
    patience=5,
    verbose=1,
    mode="min"
)
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Treniruojame modelį
print("\nPradedamas mokymas...")
history = model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[
        earlystopper,
        checkpoint,
        trainLogger,
        reduceLROnPlat,
        tb_callback,
        model2onnx
    ]
)

train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

print(f"\n✓ Mokymas baigtas!")
print(f"✓ Modelis išsaugotas: {configs.model_path}/model.h5")
print(f"✓ ONNX modelis: {configs.model_path}/model.onnx")
