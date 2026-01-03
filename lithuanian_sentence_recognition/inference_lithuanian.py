import os
import cv2
import typing
import numpy as np
import tensorflow as tf
import tf2onnx

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer  # SVARBU: Pridėta import


class LithuanianSentenceModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        """
        Atpažįsta tekstą iš vaizdo

        Args:
            image: OpenCV formatu (numpy array)

        Returns:
            str: Atpažintas tekstas
        """
        image = ImageResizer.resize_maintaining_aspect_ratio(
            image,
            *self.input_shapes[0][1:3][::-1]
        )

        # Paruošiame vaizdą modeliui
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Atliekame prognozę
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        # Dekoduojame CTC išvestį į tekstą
        text = ctc_decoder(preds, self.char_list)[0]

        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    MODEL_DIR = "Models/LT_Sentence_Recognition/202512071555"

    print("=" * 60)
    print("LIETUVIŠKŲ SAKINIŲ ATPAŽINIMO TESTAVIMAS")
    print("=" * 60)
    print(f"Modelio katalogas: {MODEL_DIR}\n")

    # Įkeliame konfigūraciją
    configs = BaseModelConfigs.load(os.path.join(MODEL_DIR, "configs.yaml"))

    print(f" Konfigūracija įkelta")
    print(f"  Žodynas: {configs.vocab}")
    print(f"  Žodyno dydis: {len(configs.vocab)}")
    print(f"  Vaizdo dydis: {configs.width}x{configs.height}\n")

    onnx_path = os.path.join(configs.model_path, "model.onnx")
    h5_path = os.path.join(configs.model_path, "model.h5")

    if not os.path.exists(onnx_path):
        print("ONNX failas nerastas. Konvertuojama iš .h5...")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 failas nerastas: {h5_path}")

        print(f"Įkeliamas H5 modelis: {h5_path}")
        model = tf.keras.models.load_model(h5_path, compile=False, safe_mode=False)

        print("Konvertuojama į ONNX...")
        spec = (tf.TensorSpec((None, configs.height, configs.width, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)

        print(f"ONNX sukurtas: {onnx_path}\n")
    else:
        print(f"ONNX modelis rastas: {onnx_path}\n")

    print("Inicializuojamas inference modelis...")
    inference_model = LithuanianSentenceModel(
        model_path=configs.model_path,
        char_list=configs.vocab
    )
    print("Modelis įkeltas sėkmingai\n")

    val_csv = os.path.join(configs.model_path, "val.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validavimo duomenys nerasti: {val_csv}")

    print(f"Kraunami validavimo duomenys: {val_csv}")
    df = pd.read_csv(val_csv).values.tolist()
    print(f"Įkelta {len(df)} pavyzdžių\n")

    print("=" * 60)
    print("PRADEDAMAS TESTAVIMAS")
    print("=" * 60)

    # Testuojame modelį
    accum_cer = []
    accum_wer = []
    correct_predictions = 0

    for image_path, label in tqdm(df, desc="Testuojama"):
        image_path = image_path.replace("\\", "/")
        image = cv2.imread(image_path)

        if image is None:
            print(f"Nepavyko nuskaityti: {image_path}")
            continue

        prediction_text = inference_model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)

        if prediction_text == label:
            correct_predictions += 1

        accum_cer.append(cer)
        accum_wer.append(wer)

        if len(accum_cer) <= 5:
            print(f"\nPavyzdys {len(accum_cer)}:")
            print(f"  Label:      {label}")
            print(f"  Prognozė:   {prediction_text}")
            print(f"  CER: {cer:.4f}, WER: {wer:.4f}")

    print("\n" + "=" * 60)
    print("TESTAVIMO REZULTATAI")
    print("=" * 60)
    print(f"Iš viso pavyzdžių: {len(df)}")
    print(f"Vidutinis CER (Character Error Rate): {np.average(accum_cer):.4f}")
    print(f"Vidutinis WER (Word Error Rate): {np.average(accum_wer):.4f}")
    print(f"Tikslumas (Exact Match): {correct_predictions}/{len(df)} ({100 * correct_predictions / len(df):.2f}%)")
    print("=" * 60)

    # Papildoma statistika
    print("\nPapildoma statistika:")
    print(
        f"  CER < 0.1: {sum(1 for c in accum_cer if c < 0.1)}/{len(accum_cer)} ({100 * sum(1 for c in accum_cer if c < 0.1) / len(accum_cer):.1f}%)")
    print(
        f"  CER < 0.2: {sum(1 for c in accum_cer if c < 0.2)}/{len(accum_cer)} ({100 * sum(1 for c in accum_cer if c < 0.2) / len(accum_cer):.1f}%)")
    print(
        f"  WER < 0.3: {sum(1 for w in accum_wer if w < 0.3)}/{len(accum_wer)} ({100 * sum(1 for w in accum_wer if w < 0.3) / len(accum_wer):.1f}%)")
    print("=" * 60)