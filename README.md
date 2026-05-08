# Lithuanian Handwriting Recognition – API

REST API for Lithuanian handwritten sentence recognition using a CNN-BiLSTM-CTC model.

Model from https://pylessons.com/handwritten-sentence-recognition

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status |
| GET | `/info` | Model metadata (vocab, image size, etc.) |
| POST | `/predict` | Recognize handwriting from a base64 image |

### POST /predict

```json
{
  "image": "<base64 encoded PNG/JPG>"
}
```

**Response:**
```json
{
  "success": true,
  "text": "recognized text"
}
```

## Switching Models

To change the active model, update `MODEL_DIR` in `api.py`:

```python
MODEL_DIR = "Models/LT_Progressive_Scratch/M2p"
```

All models are located in the `Models/` folder:

| Model | Path | Training method | Data ratio | Holdout CER |
|-------|------|-----------------|------------|-------------|
| M1 | `Models/LT_Sentence_Recognition/M1` | Standard | 1:1 | – (val_CER 0.4597) |
| M2 | `Models/LT_Sentence_Recognition/M2` | Standard | 1:2 | – (val_CER 0.2518) |
| M0p | `Models/LT_Progressive_Scratch/M0p` | Progressive | Real only | 0.6882 |
| M1p | `Models/LT_Progressive_Scratch/M1p` | Progressive | 1:1 | 0.2040 |
| M2p | `Models/LT_Progressive_Scratch/M2p` | Progressive | 1:2 | 0.2139 |

M1 and M2 were trained using standard training (200 epochs) on a mix of real and synthetic data, and evaluated on a validation set drawn from the same dataset (90/10 train/val split), so no holdout CER is reported. M0p–M2p were trained using progressive training - iteratively refining the model across 10 runs (up to 100 epochs per run) with a real+synthetic dataset at the specified ratio - and evaluated on an independent holdout set of 250 real sentences unseen during training.

## Model

CNN-BiLSTM-CTC architecture trained on Lithuanian handwritten sentences. Input image size: 199×2262px. Outputs a UTF-8 string with Lithuanian characters.

## Stack

- Python, FastAPI
- TensorFlow / Keras
- ONNX Runtime