# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

IMAGE_W    = 2262
IMAGE_H    = 199
PADDING_X  = 10
PADDING_Y  = 5
TEXT_MAX_W = IMAGE_W - 2 * PADDING_X
TEXT_MAX_H = IMAGE_H - 2 * PADDING_Y

FIXED_SENTENCES = [
    "Aš atsargiai atidarau ąžuolinę dėžę.",
    "Ąžuolas auga atokiame kaimo kampe.",
    "Vaikai atbėga iš aikštės.",
    "Mama atsisėda ant aukštos kėdės.",
    "Brolis bėga per baltą sniegą.",
    "Bibliotekoje buvo labai tylu.",
    "Berniukas beldžia į baltas duris.",
    "Čia gyvena labai draugiška šeima.",
    "Cukrų ir cinamoną dedame į pyragą.",
    "Čiulba paukščiai ankstų rytą.",
    "Cecilija čiuožia ant ledo.",
    "Daktaras dirba didelėje ligoninėje.",
    "Draugas davė dovaną per gimtadienį.",
    "Daina skamba per visą sodą.",
    "Eglė eina į egzaminą ramiai.",
    "Ežeras ežiukui atrodo labai didelis.",
    "Elnias eina per mišką tyliai.",
    "Filharmonijoje groja nuostabus orkestras.",
    "Fotografas fotografuoja gražius peizažus.",
    "Futbolininkas šokinėja per kliūtis.",
    "Filmas buvo labai įdomus ir jaudinantis.",
    "Gydytojas geranoriškai gydo ligonius.",
    "Gieda paukščiai, kol saulė leidžiasi.",
    "Gintaras groja gitara vakarais.",
    "Henrikas harfą groja labai jautriai.",
    "Horizonte šviečia aušros žara.",
    "Herojus niekada nepraranda drąsos.",
    "Ingrida ir Irena eina į kiną.",
    "Įdomus žurnalas guli ant stalo.",
    "Ypač skanūs obuoliai auga sode.",
    "Jonas į darbą važiuoja dviračiu.",
    "Ieva renka įvairius akmenis paplūdimyje.",
    "Jonas joja per žalią pievą.",
    "Jūra ošia, o žuvėdros skraido.",
    "Julija juokiasi ir šoka kartu.",
    "Katė keliasi anksti kiekvieną rytą.",
    "Knyga gulėjo ant kambario suolo.",
    "Kavinėje kvepėjo kava ir cinamonu.",
    "Kartu keliaujame per kalnų taką.",
    "Lauke lyja lietus ir pučia vėjas.",
    "Liepa lapoja, o vaikai žaidžia.",
    "Laura lipa laiptais aukštyn.",
    "Lapai krenta nuo liepos rugsėjį.",
    "Mama myli mažus vaikus.",
    "Mėnulis šviečia per debesį.",
    "Mokytojas mokiniams paaiškina matematiką.",
    "Naktį žvaigždės šviečia ryškiausiai.",
    "Namai stovi netoli ežero.",
    "Nuostabus gamtos vaizdas atsiveria nuo kalno.",
    "Neringa nusprendė nueiti į pamoką.",
    "Obelys žydi baltais žiedais pavasarį.",
    "Oras šiandien labai gražus ir šiltas.",
    "Ona išeina iš namų ankstyvą rytą.",
    "Orelis skrenda aukštai virš kalnų.",
    "Paukštis plasnoja sparnais prie upės.",
    "Parke vaikai žaidžia iki vakaro.",
    "Petras pasivaikščioja prie ežero.",
    "Rytas prasideda nuo kavos ir knygos.",
    "Raudona rožė žydi sode.",
    "Reginos ranka rašo greitai.",
    "Ruduo atneša spalvotus lapus.",
    "Saulė šviečia ir visi džiaugiasi.",
    "Šuo seka šeimininką visur.",
    "Studentas skaito storą knygą.",
    "Tėtis treniruoja vaikus kiekvieną savaitę.",
    "Tai buvo tikrai nuostabus vakaras.",
    "Tomas tvarko savo kambarį.",
    "Tolumoje matosi aukšti kalnai.",
    "Upė teka tarp ūksmingų medžių.",
    "Ūkininkas užaugino didelį ūkį.",
    "Universitete mokosi daug įdomių žmonių.",
    "Vakaras atėjo tyliai ir ramiai.",
    "Vaikas verkia, nes pametė žaislą.",
    "Vasarą važiuojame prie jūros.",
    "Vytautas visada verčia sunkias knygas.",
    "Žiema atneša sniegą ir šaltį.",
    "Žuvis plaukia giliame ežere.",
    "Zigmas žiūri žaibą per langą.",
    "Žiedai žydi žalioje pievoje.",
    "Šį rytą žvejys išplaukė anksti ir parsinešė daug žuvų.",
    "Žiemą vaikai mėgsta čiuožti ant ežero ledo.",
    "Jurga ir Žilvinas keliavo per kalnus visą savaitę.",
    "Šaltu rytu katė susirango prie šiltų durų.",
    "Ąžuolų miškas ošia, kai pučia stiprus vėjas.",
    "Žaibas nušvietė tamsų dangų audros metu.",
    "Senoji bibliotekininkė žino visų knygų vietas.",
    "Šventė prasidėjo linksmai ir baigėsi džiaugsmingai.",
    "Vaikas džiaugsmingai šokinėja per balas po lietaus.",
    "Žuvėdros skraido žemai virš bangų.",
    "Šuo išbėgo į kiemą ir pradėjo loti.",
    "Žirgai bėga laisvai per žalią lauką.",
    "Vakarinė saulė nusidažė oranžiniais ir rožiniais tonais.",
    "Švelnus vėjelis glostė žolę pievoje.",
    "Žvejys žvejojo visą naktį prie upės.",
    "Universiteto studentai žaidžia šachmatais laisvalaikiu.",
    "Šį vakarą žvaigždės ypač ryškiai šviečia.",
    "Šventinis vakaras prasidėjo su muzika ir šokiais.",
]

SUBJECTS = [
    "Jonas", "Marija", "Petras", "Eglė", "Tomas", "Laura",
    "Gintaras", "Rasa", "Mindaugas", "Jurgita", "Žilvinas", "Neringa",
    "Danielius", "Kristina", "Mantas", "Ingrida", "Saulius", "Renata",
    "mokytojas", "mokytoja", "studentas", "studentė", "gydytojas",
    "vaikas", "berniukas", "mergaitė", "katė", "šuo",
]

VERBS_INF = [
    "eiti", "bėgti", "dirbti", "mokytis", "skaityti", "rašyti",
    "kalbėti", "žaisti", "valgyti", "gerti", "šokti", "dainuoti",
    "piešti", "galvoti", "sportuoti", "keliauti", "ilsėtis",
    "tvarkytis", "plaukti", "važiuoti", "groti",
]

PLACES = [
    "mokykloje", "bibliotekoje", "parke", "mieste", "kaime",
    "miške", "paplūdimyje", "kalnuose", "sode", "kavinėje",
    "stadione", "ligoninėje", "stotyje", "namuose", "darbe",
    "pievoje", "upės pakrantėje", "ežero pakrantėje",
]

ADVERBS = [
    "Greitai", "Lėtai", "Ramiai", "Garsiai", "Tyliai",
    "Džiaugsmingai", "Rūpestingai", "Atidžiai", "Mielai",
    "Kruopščiai", "Drąsiai", "Švelniai",
]

ADJECTIVES = [
    "gražų", "didelį", "mažą", "seną", "naują",
    "įdomų", "nuobodų", "skanų", "šiltą", "šaltą",
    "greitą", "lėtą", "spalvotą", "tamsų", "šviesų",
]

TIMES = [
    "ryte", "vakare", "naktį", "po pietų",
    "anksti ryte", "vėlai vakare",
]

REASONS = [
    "nes turi daug darbo",
    "nes nori pailsėti",
    "nes tai labai įdomu",
    "nes jam patinka",
    "nes reikia pasiruošti",
]

OBJECTS_SKAITO = ["knygą", "žurnalą", "laišką", "istoriją", "receptą", "žemėlapį"]
OBJECTS_RASO   = ["laišką", "užduotį", "projektą", "istoriją", "žemėlapį", "receptą"]
OBJECTS_NESA   = ["knygą", "dovaną", "žaislą", "krepšį", "gėlę", "laišką"]
OBJECTS_PERKA  = ["knygą", "kavą", "arbatą", "duoną", "dovaną", "gėlę", "žaislą"]
OBJECTS_RENKA  = ["gėles", "obuolius", "daržoves", "vaisius", "akmenis", "grybus"]
OBJECTS_GERIA  = ["kavą", "arbatą", "pieną", "sultis", "vandenį"]
OBJECTS_VALGO  = ["duoną", "daržoves", "vaisius", "košę", "sriubą", "obuolį"]

TEMPLATES = [
    ("{subj} ilsisi {place}.",                   lambda: {}),
    ("{subj} dirba {place}.",                    lambda: {}),
    ("{subj} mokosi {place}.",                   lambda: {}),
    ("{subj} sportuoja {place}.",                lambda: {}),
    ("{subj} žaidžia {place}.",                  lambda: {}),
    ("{subj} gyvena {place}.",                   lambda: {}),
    ("{subj} lankosi {place}.",                  lambda: {}),
    ("{subj} skaito {obj}.",        lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} rašo {obj}.",          lambda: {"obj": random.choice(OBJECTS_RASO)}),
    ("{subj} neša {obj}.",          lambda: {"obj": random.choice(OBJECTS_NESA)}),
    ("{subj} perka {obj}.",         lambda: {"obj": random.choice(OBJECTS_PERKA)}),
    ("{subj} renka {obj}.",         lambda: {"obj": random.choice(OBJECTS_RENKA)}),
    ("{subj} geria {obj}.",         lambda: {"obj": random.choice(OBJECTS_GERIA)}),
    ("{subj} valgo {obj}.",         lambda: {"obj": random.choice(OBJECTS_VALGO)}),
    ("{subj} nori {inf}.",                       lambda: {}),
    ("{subj} mėgsta {inf}.",                     lambda: {}),
    ("{subj} stengiasi {inf}.",                  lambda: {}),
    ("{subj} pradeda {inf}.",                    lambda: {}),
    ("{subj} bando {inf}.",                      lambda: {}),
    ("{subj} nori skaityti {obj}.",  lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} mėgsta skaityti {obj}.",lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} nori rašyti {obj}.",    lambda: {"obj": random.choice(OBJECTS_RASO)}),
    ("{subj} nori pirkti {obj}.",    lambda: {"obj": random.choice(OBJECTS_PERKA)}),
    ("{subj} mėgsta pirkti {obj}.",  lambda: {"obj": random.choice(OBJECTS_PERKA)}),
    ("{subj} nori valgyti {obj}.",   lambda: {"obj": random.choice(OBJECTS_VALGO)}),
    ("{subj} mėgsta valgyti {obj}.", lambda: {"obj": random.choice(OBJECTS_VALGO)}),
    ("{subj} nori gerti {obj}.",     lambda: {"obj": random.choice(OBJECTS_GERIA)}),
    ("{subj} mėgsta gerti {obj}.",   lambda: {"obj": random.choice(OBJECTS_GERIA)}),
    ("{adv} {subj} dirba {place}.",              lambda: {}),
    ("{adv} {subj} ilsisi {place}.",             lambda: {}),
    ("{adv} {subj} mokosi {place}.",             lambda: {}),
    ("{adv} {subj} skaito {obj}.",  lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{adv} {subj} rašo {obj}.",    lambda: {"obj": random.choice(OBJECTS_RASO)}),
    ("{adv} {subj} neša {obj}.",    lambda: {"obj": random.choice(OBJECTS_NESA)}),
    ("{adv} {subj} perka {obj}.",   lambda: {"obj": random.choice(OBJECTS_PERKA)}),
    ("{adv} {subj} valgo {obj}.",   lambda: {"obj": random.choice(OBJECTS_VALGO)}),
    ("Šiandien {subj} dirba {place}.",           lambda: {}),
    ("Šiandien {subj} mokosi {place}.",          lambda: {}),
    ("Šiandien {subj} ilsisi {place}.",          lambda: {}),
    ("Šiandien {subj} skaito {obj}.",lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Šiandien {subj} perka {obj}.", lambda: {"obj": random.choice(OBJECTS_PERKA)}),
    ("Vakar {subj} dirbo {place}.",              lambda: {}),
    ("Vakar {subj} skaitė {obj}.",  lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Vakar {subj} nešė {obj}.",    lambda: {"obj": random.choice(OBJECTS_NESA)}),
    ("Rytoj {subj} eis {place}.",                lambda: {}),
    ("Rytoj {subj} skaitys {obj}.", lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Ar {subj} dirba {place}?",                 lambda: {}),
    ("Ar {subj} mokosi {place}?",                lambda: {}),
    ("Ar {subj} skaito {obj}?",     lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Ar {subj} mėgsta {inf}?",                  lambda: {}),
    ("Ar {subj} nori {inf}?",                    lambda: {}),
    ("Kodėl {subj} dirba {place}?",              lambda: {}),
    ("Kur {subj} dirba?",                        lambda: {}),
    ("Ką {subj} skaito?",                        lambda: {}),
    ("{subj} skaito {adj} {obj}.",
     lambda: {"obj": random.choice(OBJECTS_SKAITO), "adj": random.choice(ADJECTIVES)}),
    ("{subj} perka {adj} {obj}.",
     lambda: {"obj": random.choice(OBJECTS_PERKA),  "adj": random.choice(ADJECTIVES)}),
    ("{subj} dirba {place} {time}.",
     lambda: {"time": random.choice(TIMES)}),
    ("{subj} mokosi {place} {time}.",
     lambda: {"time": random.choice(TIMES)}),
    ("{subj} dirba {place}, {reason}.",
     lambda: {"reason": random.choice(REASONS)}),
    ("{subj} mokosi {place}, {reason}.",
     lambda: {"reason": random.choice(REASONS)}),
    ("{subj} dirba {place} ir skaito {obj}.",
     lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} mokosi {place}, bet nori {inf}.",   lambda: {}),
    ("{subj} nedirba {place}.",                  lambda: {}),
    ("{subj} neskaito {obj}.",
     lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} nenori {inf}.",                     lambda: {}),
    ("{adv} {subj} skaito {obj} {place}.",
     lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} nori {inf}, nes {reason}.",
     lambda: {"reason": random.choice(REASONS)}),
    ("{adv} {subj} mokosi {place} {time}.",
     lambda: {"time": random.choice(TIMES)}),
]


def generate_random_sentence() -> str:
    tmpl, extras_fn = random.choice(TEMPLATES)
    extras = extras_fn()
    base_args = dict(
        subj  = random.choice(SUBJECTS),
        inf   = random.choice(VERBS_INF),
        place = random.choice(PLACES),
        adv   = random.choice(ADVERBS),
    )
    base_args.update(extras)
    s = tmpl.format(**base_args)
    return s[0].upper() + s[1:]


def generate_sentences_pool(n: int) -> list:
    pool = FIXED_SENTENCES.copy()
    while len(pool) < n:
        s = generate_random_sentence()
        if len(s) <= 45:
            pool.append(s)
    random.shuffle(pool)
    return pool[:n]

def supports_all_chars(font_path: str, text: str) -> bool:
    try:
        font = ImageFont.truetype(font_path, 32)
        for char in text:
            if char == ' ':
                continue
            if not font.getmask(char).getbbox():
                return False
        return True
    except Exception:
        return False

def apply_augmentations(img_np: np.ndarray) -> np.ndarray:
    if random.random() < 0.6:
        sigma = random.uniform(3, 12)
        noise = np.random.normal(0, sigma, img_np.shape)
        img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        ksize = random.choice([3, 3, 5])
        img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta  = random.uniform(-15, 15)
        img_np = np.clip(img_np.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        angle = random.uniform(-1.5, 1.5)
        h, w = img_np.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_np = cv2.warpAffine(img_np, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)

    if random.random() < 0.25:
        h, w = img_np.shape
        d = random.randint(3, 8)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.randint(0, d), random.randint(0, d)],
            [w - random.randint(0, d), random.randint(0, d)],
            [random.randint(0, d), h - random.randint(0, d)],
            [w - random.randint(0, d), h - random.randint(0, d)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
    return img_np

def render_sentence_image(sentence: str, font_path: str) -> Image.Image:
    font_size = 100
    while font_size > 20:
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(sentence)
        if (bbox[2] - bbox[0]) <= TEXT_MAX_W and (bbox[3] - bbox[1]) <= TEXT_MAX_H:
            break
        font_size -= 2

    bbox   = font.getbbox(sentence)
    text_h = bbox[3] - bbox[1]
    x = PADDING_X - bbox[0]
    y = (IMAGE_H - text_h) // 2 - bbox[1] + random.randint(-3, 3)

    img  = Image.new("L", (IMAGE_W, IMAGE_H), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), sentence, fill=random.randint(20, 80), font=font)
    return img

def generate_dataset(output_dir: str = "Datasets/Synthetic_Sentences",
                     num_samples: int = 70000,
                     fonts: list = None) -> list:
    if fonts is None:
        fonts = [
            "Fonts/HomemadeApple-Regular.ttf",
            "Fonts/ReenieBeanie-Regular.ttf",
            "Fonts/IndieFlower-Regular.ttf",
        ]
    fonts = [f for f in fonts if os.path.exists(f)]
    if not fonts:
        raise RuntimeError("Nerasta ne vieno srifto kataloge 'Fonts/'")

    print(f"Naudojami sriftai: {fonts}")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    sentences   = generate_sentences_pool(num_samples)
    annotations = []

    print(f"Generuojama {num_samples} vaizdu ({IMAGE_W}x{IMAGE_H}px)...")

    for i, sentence in enumerate(sentences):
        valid_fonts = [f for f in fonts if supports_all_chars(f, sentence)]
        if not valid_fonts:
            continue
        try:
            img = render_sentence_image(sentence, random.choice(valid_fonts))
        except Exception as e:
            print(f"  Klaida: '{sentence}': {e}")
            continue

        img_np   = apply_augmentations(np.array(img))
        filepath = os.path.join(images_dir, f"sentence_{i:05d}.png")
        Image.fromarray(img_np).save(filepath)
        annotations.append([filepath, sentence])

        if (i + 1) % 1000 == 0:
            print(f"  Sugeneruota: {i + 1}/{num_samples}")

    csv_path = os.path.join(output_dir, "annotations.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for filepath, sentence in annotations:
            f.write(f"{filepath},{sentence}\n")

    print(f"\nBaigta. Sugeneruota: {len(annotations)} vaizdu")
    print(f"Vaizdai:    {images_dir}")
    print(f"Anotacijos: {csv_path}")
    return annotations


if __name__ == "__main__":
    fonts = [
        "Fonts/HomemadeApple-Regular.ttf",
        "Fonts/ReenieBeanie-Regular.ttf",
        "Fonts/IndieFlower-Regular.ttf",
    ]
    test_chars = " aąbcčdeęėfghiįyjklmnoprsštuųūvzžAĄBCČDEĘĖFGHIĮYJKLMNOPRSŠTUŲŪVZŽ.,!?-"
    for font_path in fonts:
        if not os.path.exists(font_path):
            print(f"Sriftas nerastas: {font_path}")
            continue
        missing = [c for c in test_chars if not supports_all_chars(font_path, c)]
        if missing:
            print(f"{font_path}: Truksta simboliu: {missing}")
        else:
            print(f"OK {font_path}: palaiko visus simbolius")

    generate_dataset(output_dir="Datasets/Synthetic_Sentences", num_samples=70000)