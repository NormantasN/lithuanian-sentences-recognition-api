import os
import base64
import tensorflow as tf
import tf2onnx
from flask import Flask, request, jsonify
from flask_cors import CORS
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

import numpy as np
import cv2

MODEL_DIR = "Models/LT_Sentence_Recognition/202512071555"

app = Flask(__name__)
CORS(app)

model = None
configs = None


class LithuanianSentenceModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        height, width = image.shape[:2]
        target_width = self.input_shapes[0][2]
        target_height = self.input_shapes[0][1]

        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(image, (new_width, new_height))

        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded[:new_height, :new_width] = image

        image_batch = np.expand_dims(padded, axis=0).astype(np.float32)

        preds = self.model.run(
            self.output_names,
            {self.input_names[0]: image_batch}
        )[0]

        return ctc_decoder(preds, self.char_list)[0]


def load_model(model_dir):
    global model, configs

    print("=" * 60)
    print("LIETUVIŠKŲ SAKINIŲ ATPAŽINIMO API")
    print("=" * 60)
    print(f"Modelio katalogas: {model_dir}\n")

    configs_path = os.path.join(model_dir, "configs.yaml")
    print("DEBUG configs_path exists:", os.path.exists(configs_path))

    # DEBUG YAML load
    import yaml
    with open(configs_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    print("DEBUG YAML data:", data)


    configs = BaseModelConfigs.load(configs_path)
    if configs is None:
        raise RuntimeError(f"Failed to load configs from {configs_path}")
    print(f"Konfigūracija įkelta")
    print(f"  Žodynas: {configs.vocab}")
    print(f"  Žodyno dydis: {len(configs.vocab)}")
    print(f"  Vaizdo dydis: {configs.width}x{configs.height}\n")

    onnx_path = os.path.join(configs.model_path, "model.onnx")
    h5_path = os.path.join(configs.model_path, "model.h5")
    print("DEBUG cwd:", os.getcwd())
    print("DEBUG model_path:", configs.model_path)
    print("DEBUG h5 exists:", os.path.exists(h5_path))
    print("DEBUG onnx exists:", os.path.exists(onnx_path))

    if not os.path.exists(onnx_path):
        print("ONNX failas nerastas. Konvertuojama iš .h5...")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 failas nerastas: {h5_path}")

        print(f"Įkeliamas H5 modelis: {h5_path}")
        keras_model = tf.keras.models.load_model(h5_path, compile=False)

        print("Konvertuojama į ONNX...")
        spec = (tf.TensorSpec((None, configs.height, configs.width, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, output_path=onnx_path)

        print(f"ONNX sukurtas: {onnx_path}\n")
    else:
        print(f"ONNX modelis rastas: {onnx_path}\n")

    print("Inicializuojamas inference modelis...")
    model = LithuanianSentenceModel(
        model_path=configs.model_path,
        char_list=configs.vocab
    )
    print("Modelis įkeltas sėkmingai!\n")


def decode_image(image_data):
    if isinstance(image_data, str):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Nepavyko dekoduoti vaizdo")

    return image


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': configs.model_path if configs else None
    })


@app.route('/info', methods=['GET'])
def model_info():
    if model is None or configs is None:
        return jsonify({
            'success': False,
            'error': 'Modelis neįkeltas'
        }), 500

    return jsonify({
        'success': True,
        'vocabulary': configs.vocab,
        'vocab_size': len(configs.vocab),
        'max_text_length': configs.max_text_length,
        'image_height': configs.height,
        'image_width': configs.width
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Atpažįsta tekstą iš vaizdo

    Request body (JSON):
        {
            "image": "base64_encoded_image"
        }

    OR multipart/form-data:
        file: image file

    Response:
        {
            "success": true,
            "text": "atpažintas tekstas"
        }
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Modelis neįkeltas'
        }), 500

    try:
        # Tikriname ar yra failas (multipart/form-data)
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
            image = decode_image(image_bytes)

        # Arba JSON su base64 (application/json)
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Trūksta "image" lauko request body'
                }), 400

            image = decode_image(data['image'])

        else:
            return jsonify({
                'success': False,
                'error': 'Netinkamas request formatas. Naudokite JSON su "image" lauku arba multipart/form-data su "file"'
            }), 400

        # Atliekame prognozę
        predicted_text = model.predict(image)

        return jsonify({
            'success': True,
            'text': predicted_text
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Atpažįsta tekstą keliuose vaizduose

    Request body (JSON):
        {
            "images": ["base64_image1", "base64_image2", ...]
        }

    Response:
        {
            "success": true,
            "results": [
                {"index": 0, "text": "tekstas1", "success": true},
                {"index": 1, "text": "tekstas2", "success": true}
            ]
        }
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Modelis neįkeltas'
        }), 500

    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request turi būti JSON formatu'
            }), 400

        data = request.get_json()
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({
                'success': False,
                'error': 'Trūksta "images" sąrašo'
            }), 400

        results = []
        for i, image_data in enumerate(data['images']):
            try:
                image = decode_image(image_data)
                predicted_text = model.predict(image)
                results.append({
                    'index': i,
                    'text': predicted_text,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'success': False
                })

        return jsonify({
            'success': True,
            'results': results,
            'total': len(data['images']),
            'successful': sum(1 for r in results if r['success'])
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': repr(e)
        }), 500


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Lietuviškų sakinių atpažinimo API serveris')
#     parser.add_argument('--model_path', type=str, default=MODEL_DIR,
#                         help='Kelias iki modelio katalogo')
#     parser.add_argument('--host', type=str, default='0.0.0.0',
#                         help='Serverio adresas (default: 0.0.0.0)')
#     parser.add_argument('--port', type=int, default=8000,
#                         help='Serverio portas (default: 8000)')
#     parser.add_argument('--debug', action='store_true',
#                         help='Debug režimas')
#
#     args = parser.parse_args()
#
#     # Įkeliame modelį
#     try:
#         load_model(args.model_path)
#     except Exception as e:
#         print(f"Nepavyko įkelti modelio: {e}")
#         import traceback
#
#         traceback.print_exc()
#         exit(1)
#
#     print("=" * 60)
#     print(f"API serveris veikia: http://{args.host}:{args.port}")
#     print("=" * 60)
#     print("\n Galimi endpoint'ai:")
#     print(f"  GET  /health         - Serverio būsenos tikrinimas")
#     print(f"  GET  /info           - Modelio informacija")
#     print(f"  POST /predict        - Atpažinti vieną vaizdą")
#     print(f"  POST /batch_predict  - Atpažinti kelis vaizdus")
#     print("=" * 60 + "\n")
#
#     # Paleidžiame serverį
#     app.run(host=args.host, port=args.port, debug=args.debug)

load_model(MODEL_DIR)