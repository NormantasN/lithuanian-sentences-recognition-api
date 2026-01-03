import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def supports_all_chars(font_path, text):
    try:
        font = ImageFont.truetype(font_path, 32)
        for char in text:
            if char == ' ':
                continue
            if not font.getmask(char).getbbox():
                return False
        return True
    except:
        return False


def apply_augmentations(img_np):
    # Triukšmas
    if random.random() < 0.5:
        noise = np.random.normal(0, 10, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)

    # Blur
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

    # Perspektyva
    if random.random() < 0.3:
        h, w = img_np.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        delta = 10
        pts2 = np.float32([
            [random.randint(0, delta), random.randint(0, delta)],
            [w - random.randint(0, delta), random.randint(0, delta)],
            [random.randint(0, delta), h - random.randint(0, delta)],
            [w - random.randint(0, delta), h - random.randint(0, delta)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, M, (w, h), borderValue=255)

    return img_np


def generate_lithuanian_sentences(num_sentences=1000):
    """Generuoja atsitiktinius lietuviškus sakinius"""

    subjects = [
        "aš", "tu", "jis", "ji", "mes", "jūs", "jie", "jos",
        "vaikas", "mergaitė", "moteris", "vyras", "mokytojas", "studentas",
        "gydytojas", "draugas", "šuo", "katė", "paukštis"
    ]

    verbs = [
        "eina", "bėga", "dirba", "mokosi", "skaito", "rašo", "kalba", "žaidžia",
        "valgo", "geria", "miega", "šoka", "dainuoja", "piešia", "galvoja",
        "mąsto", "juokiasi", "verkia", "šypsosi", "žiūri", "klauso"
    ]

    objects = [
        "knygą", "namus", "mokyklą", "darbą", "filmą", "muziką", "maistą",
        "vandenį", "kavą", "arbatą", "paveikslą", "daržoves", "vaisius",
        "žaidimą", "pamoką", "egzaminą", "projektą", "užduotį"
    ]

    adjectives = [
        "gražus", "didelis", "mažas", "geras", "blogas", "naujas", "senas",
        "greitas", "lėtas", "šiltas", "šaltas", "lengvas", "sunkus", "žalias",
        "mėlynas", "raudonas", "juodas", "baltas", "linksmas", "liūdnas"
    ]

    adverbs = [
        "greitai", "lėtai", "gerai", "blogai", "dabar", "vakar", "rytoj",
        "šiandien", "visada", "niekada", "dažnai", "retai", "šiek tiek",
        "labai", "per daug", "smagiai", "ramiai", "tyliai", "garsiai"
    ]

    places = [
        "namuose", "mokykloje", "darbe", "parke", "mieste", "kaime",
        "miške", "paplūdimyje", "kalnuose", "prie ežero", "sode", "bibliotekoje"
    ]

    patterns = [
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(places)}.",
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}.",
        lambda: f"{random.choice(adjectives)} {random.choice(subjects)} {random.choice(verbs)}.",
        lambda: f"{random.choice(subjects)} {random.choice(adverbs)} {random.choice(verbs)}.",
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(adjectives)} {random.choice(objects)}.",
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} ir {random.choice(verbs)}.",
        lambda: f"ar {random.choice(subjects)} {random.choice(verbs)}?",
        lambda: f"{random.choice(subjects)} labai {random.choice(verbs)} {random.choice(places)}.",
        lambda: f"{random.choice(subjects)} nori {random.choice(verbs)} {random.choice(objects)}.",
        lambda: f"šiandien {random.choice(subjects)} {random.choice(verbs)} {random.choice(adverbs)}.",
    ]

    sentences = []
    for _ in range(num_sentences):
        pattern = random.choice(patterns)
        sentence = pattern()
        # Pirma raidė didžioji
        sentence = sentence[0].upper() + sentence[1:]
        sentences.append(sentence)

    return sentences


def generate_dataset(output_dir="Datasets/Test_Sentences",
                     sentences=None,
                     image_size=(1408, 96),
                     font_size=48,
                     num_samples=10000):

    if sentences is None:
        sentences = generate_lithuanian_sentences(num_samples)

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    annotations = []

    fonts = [
        "Fonts/HomemadeApple-Regular.ttf",
        "Fonts/ReenieBeanie-Regular.ttf",
        "Fonts/IndieFlower-Regular.ttf",
        "Fonts/Caveat-VariableFont_wght.ttf",
        "Fonts/PatrickHand-Regular.ttf",
        "Fonts/PermanentMarker-Regular.ttf"
    ]
    fonts = [f for f in fonts if os.path.exists(f)]
    if not fonts:
        raise RuntimeError("Nerasta nė vieno šrifto kataloge 'Fonts/'")

    print(f"Generuojama {len(sentences)} sakinių...")

    for i, sentence in enumerate(sentences):
        # Filtruojame šriftus, palaikančius visus sakinio simbolius
        valid_fonts = [f for f in fonts if supports_all_chars(f, sentence)]
        if not valid_fonts:
            print(f"Perspėjimas: nerastas tinkamas šriftas sakiniui: {sentence}")
            continue

        font_path = random.choice(valid_fonts)

        # Pritaikome šrifto dydį pagal sakinio ilgį
        estimated_width = len(sentence) * font_size * 0.6
        if estimated_width > image_size[0]:
            adjusted_font_size = int(font_size * (image_size[0] / estimated_width) * 0.9)
        else:
            adjusted_font_size = font_size

        font = ImageFont.truetype(font_path, adjusted_font_size)

        # Sukuriame baltą foną su juodu tekstu
        img = Image.new("L", image_size, color=255)
        draw = ImageDraw.Draw(img)

        # Skaičiuojame teksto dydį ir poziciją
        bbox = draw.textbbox((0, 0), sentence, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Centruojame tekstą
        pos = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
        draw.text(pos, sentence, fill=0, font=font)

        # PIL -> numpy
        img_np = np.array(img)

        # Pridedame augmentacijas
        img_np = apply_augmentations(img_np)

        # Numpy -> PIL
        img_aug = Image.fromarray(img_np)

        filename = f"sentence_{i:05d}.png"
        filepath = os.path.join(images_dir, filename)
        img_aug.save(filepath)

        annotations.append([filepath, sentence])

        if (i + 1) % 1000 == 0:
            print(f"Sugeneruota: {i + 1}/{len(sentences)}...")

    print(f"\nBaigta! Iš viso sugeneruota: {len(annotations)} įrašų.")
    print(f"Duomenys išsaugoti: {output_dir}")

    # Išsaugome anotacijas CSV formatu
    csv_path = os.path.join(output_dir, "annotations.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        for filepath, sentence in annotations:
            f.write(f"{filepath},{sentence}\n")

    print(f"Anotacijos išsaugotos: {csv_path}")

    return annotations


def test_font_support():
    fonts = [
        "Fonts/ReenieBeanie-Regular.ttf",
        "Fonts/IndieFlower-Regular.ttf",
        "Fonts/Caveat-VariableFont_wght.ttf",
        "Fonts/PatrickHand-Regular.ttf"
    ]

    test_chars = " aąbcčdeęėfghiįyjklmnoprsštuųūvzžAĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ.,!?-"

    for font_path in fonts:
        if not os.path.exists(font_path):
            print(f"Šriftas nerastas: {font_path}")
            continue

        missing = []
        for char in test_chars:
            if not supports_all_chars(font_path, char):
                missing.append(char)

        if missing:
            print(f"{font_path}: Trūksta {missing}")
        else:
            print(f"{font_path}: Palaiko visus simbolius")


test_font_support()

if __name__ == "__main__":
    annotations = generate_dataset(
        output_dir="Datasets/Test_Sentences",
        num_samples=10000,
        image_size=(1408, 96),
        font_size=48
    )

    print("\nPirmieji 5 pavyzdžiai:")
    for i, (path, sentence) in enumerate(annotations[:5]):
        print(f"{i + 1}. {sentence}")