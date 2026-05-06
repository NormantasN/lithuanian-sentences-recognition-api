# -*- coding: utf-8 -*-
import os
import random

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_PATH = "Fonts/dejavu-sans.book.ttf"
FONT_BOLD_PATH = "Fonts/dejavu-sans.bold.ttf"

try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', FONT_PATH))
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', FONT_BOLD_PATH))
    FONT_REGULAR = 'DejaVuSans'
    FONT_BOLD = 'DejaVuSans-Bold'
except:
    FONT_REGULAR = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'

FIXED_SENTENCES = [
    "Aš atsargiai atidarau ąžuolinę dėžę.",
    "Ąžuolas auga atokiame kaimo kampe.",
    "Vaikai atbėga iš aikštės.",
    "Mama atsisėda ant aukštos kėdės.",
    "Anas vyras atneša arbatos.",
    "Brolis bėga per baltą sniegą.",
    "Bibliotekoje buvo labai tylu.",
    "Bitė skraido virš bijūnų.",
    "Berniukas beldžia į baltas duris.",
    "Buvo labai šalta tą žiemos rytą.",
    "Čia gyvena labai draugiška šeima.",
    "Cukrų ir cinamoną dedame į pyragą.",
    "Čiulba paukščiai ankstų rytą.",
    "Cecilija čiuožia ant ledo.",
    "Čempionas treniruojasi kiekvieną dieną.",
    "Daktaras dirba didelėje ligoninėje.",
    "Draugas davė dovaną per gimtadienį.",
    "Duonos kepykloje sklinda malonus kvapas.",
    "Daina skamba per visą sodą.",
    "Danielius dažnai dalyvauja varžybose.",
    "Eglė eina į egzaminą ramiai.",
    "Ežeras ežiukui atrodo labai didelis.",
    "Ėmė lyti stiprus lietus.",
    "Elnias eina per mišką tyliai.",
    "Emilija eina į pamoką.",
    "Filharmonijoje groja nuostabus orkestras.",
    "Fotografas fotografuoja gražius peizažus.",
    "Futbolininkas šokinėja per kliūtis.",
    "Fėja iš pasakos nešioja žydrus sparnus.",
    "Filmas buvo labai įdomus ir jaudinantis.",
    "Gydytojas geranoriškai gydo ligonius.",
    "Grybų miške randa vaikai ir suaugusieji.",
    "Gieda paukščiai, kol saulė leidžiasi.",
    "Gatvėje gyveno linksmas katinas.",
    "Gintaras groja gitara vakarais.",
    "Henrikas harfą groja labai jautriai.",
    "Horizonte šviečia aušros žara.",
    "Hibiskusas žydi ryškiomis rožinėmis žiedais.",
    "Herojus niekada nepraranda drąsos.",
    "Ingrida ir Irena eina į kiną.",
    "Įdomus žurnalas guli ant stalo.",
    "Ypač skanūs obuoliai auga sode.",
    "Jonas į darbą važiuoja dviračiu.",
    "Ieva renka įvairius akmenis paplūdimyje.",
    "Jonas joja per žalią pievą.",
    "Jūra ošia, o žuvėdros skraido.",
    "Ją labai džiugino jaukūs namai.",
    "Jaunas studentas juda drąsiai į priekį.",
    "Julija juokiasi ir šoka kartu.",
    "Katė keliasi anksti kiekvieną rytą.",
    "Knyga gulėjo ant kambario suolo.",
    "Kavinėje kvepėjo kava ir cinamonu.",
    "Kartu keliaujame per kalnų taką.",
    "Kristina kraunasi daiktus prieš kelionę.",
    "Lauke lyja lietus ir pučia vėjas.",
    "Liepa lapoja, o vaikai žaidžia.",
    "Laivavedys laukia pietiniame uoste.",
    "Laura lipa laiptais aukštyn.",
    "Lapai krenta nuo liepos rugsėjį.",
    "Mama myli mažus vaikus.",
    "Mokykla yra miestelio viduryje.",
    "Mėnulis šviečia per debesį.",
    "Mokytojas mokiniams paaiškina matematiką.",
    "Mielas draugas mane sutiko prie mokyklos.",
    "Naktį žvaigždės šviečia ryškiausiai.",
    "Namai stovi netoli ežero.",
    "Nuostabus gamtos vaizdas atsiveria nuo kalno.",
    "Neringa nusprendė nueiti į pamoką.",
    "Obelys žydi baltais žiedais pavasarį.",
    "Oras šiandien labai gražus ir šiltas.",
    "Ona išeina iš namų ankstyvą rytą.",
    "Paukštis plasnoja sparnais prie upės.",
    "Parke vaikai žaidžia iki vakaro.",
    "Petras pasivaikščioja prie ežero.",
    "Rytas prasideda nuo kavos ir knygos.",
    "Rūpestinga mama rūpinasi vaikais.",
    "Raudona rožė žydi sode.",
    "Reginos ranka rašo greitai.",
    "Ruduo atneša spalvotus lapus.",
    "Saulė šviečia ir visi džiaugiasi.",
    "Šuo seka šeimininką visur.",
    "Studentas skaito storą knygą.",
    "Šiandien šalta, bet sausa.",
    "Sara šoka labai gražiai ir greitai.",
    "Tėtis treniruoja vaikus kiekvieną savaitę.",
    "Tai buvo tikrai nuostabus vakaras.",
    "Tomas tvarko savo kambarį.",
    "Upė teka tarp ūksmingų medžių.",
    "Ūkininkas užaugino didelį ūkį.",
    "Ugnis dega laužavietėje.",
    "Universitete mokosi daug įdomių žmonių.",
    "Vakaras atėjo tyliai ir ramiai.",
    "Vaikas verkia, nes pametė žaislą.",
    "Vasarą važiuojame prie jūros.",
    "Vytautas visada verčia sunkias knygas.",
    "Žiema atneša sniegą ir šaltį.",
    "Žuvis plaukia giliame ežere.",
    "Žmonės žaidžia žolėje.",
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
]

SUBJECTS = [
    "Jonas", "Marija", "Petras", "Eglė", "Tomas", "Laura",
    "Gintaras", "Rasa", "Mindaugas", "Jurgita", "Žilvinas", "Neringa",
    "Danielius", "Kristina", "Mantas", "Ingrida", "Saulius", "Renata",
    "mokytojas", "mokytoja", "studentas", "studentė", "gydytojas",
    "vaikas", "berniukas", "mergaitė", "žmogus", "šeima",
    "katė", "šuo", "paukštis", "žirgas",
]

VERBS_INF = [
    "eiti", "bėgti", "dirbti", "mokytis", "skaityti", "rašyti",
    "kalbėti", "žaisti", "valgyti", "gerti", "šokti", "dainuoti",
    "piešti", "galvoti", "sportuoti", "keliauti", "ilsėtis",
    "tvarkytis", "plaukti", "važiuoti", "groti", "statyti",
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

OBJECTS_SKAITO = ["knygą", "žurnalą", "laišką", "istoriją", "receptą", "žemėlapį"]
OBJECTS_RASO = ["laišką", "užduotį", "projektą", "istoriją", "žemėlapį", "receptą"]
OBJECTS_NESA = ["knygą", "dovaną", "žaislą", "krepšį", "gėlę", "laišką"]
OBJECTS_PERKA = ["knygą", "kavą", "arbatą", "duoną", "dovaną", "gėlę", "žaislą"]
OBJECTS_RENKA = ["gėles", "obuolius", "daržoves", "vaisius", "akmenis", "grybus"]
OBJECTS_GIRIA = ["kavą", "arbatą", "pieną", "sultis", "vandenį"]
OBJECTS_VALGO = ["duoną", "daržoves", "vaisius", "košę", "sriubą", "obuolį"]
OBJECTS_STATO = ["namą", "tvorą", "suolą", "bokštą"]
OBJECTS_GROJA = ["dainą", "melodiją", "muzikos kūrinį", "simfoniją"]
OBJECTS_INF = ["knygą", "žurnalą", "laišką", "dainą", "paveikslą",
               "kavą", "arbatą", "duoną", "dovaną", "žaislą",
               "receptą", "žemėlapį", "istoriją", "užduotį", "gėlę"]

TEMPLATES = [
    ("{subj} ilsisi {place}.", lambda: {}),
    ("{subj} dirba {place}.", lambda: {}),
    ("{subj} mokosi {place}.", lambda: {}),
    ("{subj} sportuoja {place}.", lambda: {}),
    ("{subj} žaidžia {place}.", lambda: {}),
    ("{subj} gyvena {place}.", lambda: {}),
    ("{subj} skaito {obj}.", lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{subj} rašo {obj}.", lambda: {"obj": random.choice(OBJECTS_RASO)}),
    ("{subj} neša {obj}.", lambda: {"obj": random.choice(OBJECTS_NESA)}),
    ("{subj} perka {obj}.", lambda: {"obj": random.choice(OBJECTS_PERKA)}),
    ("{subj} renka {obj}.", lambda: {"obj": random.choice(OBJECTS_RENKA)}),
    ("{subj} geria {obj}.", lambda: {"obj": random.choice(OBJECTS_GIRIA)}),
    ("{subj} valgo {obj}.", lambda: {"obj": random.choice(OBJECTS_VALGO)}),
    ("{subj} stato {obj}.", lambda: {"obj": random.choice(OBJECTS_STATO)}),
    ("{subj} groja {obj}.", lambda: {"obj": random.choice(OBJECTS_GROJA)}),
    ("{subj} nori {inf}.", lambda: {}),
    ("{subj} mėgsta {inf}.", lambda: {}),
    ("{subj} pradeda {inf}.", lambda: {}),
    ("{subj} nori {inf} {obj}.", lambda: {"obj": random.choice(OBJECTS_INF)}),
    ("{subj} mėgsta {inf} {obj}.", lambda: {"obj": random.choice(OBJECTS_INF)}),
    ("{adv} {subj} dirba {place}.", lambda: {}),
    ("{adv} {subj} mokosi {place}.", lambda: {}),
    ("{adv} {subj} skaito {obj}.", lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("{adv} {subj} rašo {obj}.", lambda: {"obj": random.choice(OBJECTS_RASO)}),
    ("{adv} {subj} neša {obj}.", lambda: {"obj": random.choice(OBJECTS_NESA)}),
    ("Šiandien {subj} dirba {place}.", lambda: {}),
    ("Šiandien {subj} mokosi {place}.", lambda: {}),
    ("Šiandien {subj} skaito {obj}.", lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Vakar {subj} dirbo {place}.", lambda: {}),
    ("Vakar {subj} skaitė {obj}.", lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Rytoj {subj} eis {place}.", lambda: {}),
    ("Ar {subj} dirba {place}?", lambda: {}),
    ("Ar {subj} skaito {obj}?", lambda: {"obj": random.choice(OBJECTS_SKAITO)}),
    ("Ar {subj} mėgsta {inf}?", lambda: {}),
]

LT_ALPHABET = "aąbcčdeęėfghiįyjklmnoprsštuųūvzž"


def sentences_cover_alphabet(sentences: list) -> bool:
    combined = "".join(sentences).lower()
    return all(c in combined for c in LT_ALPHABET)


def generate_random_sentence() -> str:
    tmpl, extras_fn = random.choice(TEMPLATES)
    extras = extras_fn()
    base_args = dict(
        subj=random.choice(SUBJECTS),
        inf=random.choice(VERBS_INF),
        place=random.choice(PLACES),
        adv=random.choice(ADVERBS),
    )
    base_args.update(extras)
    s = tmpl.format(**base_args)
    return s[0].upper() + s[1:]


def generate_lithuanian_sentences(num_sentences: int = 1000) -> list:
    base = FIXED_SENTENCES.copy()
    while len(base) < num_sentences:
        base.append(generate_random_sentence())
    random.shuffle(base)
    return base[:num_sentences]


def pick_five_covering_alphabet(pool: list, max_tries: int = 500) -> list:
    for _ in range(max_tries):
        chosen = random.sample(pool, min(5, len(pool)))
        if sentences_cover_alphabet(chosen):
            return chosen

    chosen = random.sample(pool, min(4, len(pool)))
    combined = "".join(chosen).lower()
    missing = [c for c in LT_ALPHABET if c not in combined]

    for letter in missing:
        candidates = [s for s in FIXED_SENTENCES if letter in s.lower()]
        if candidates:
            replacement = random.choice(candidates)
            if replacement not in chosen:
                chosen.append(replacement)
                if len(chosen) == 5:
                    break

    while len(chosen) < 5:
        extra = random.choice(pool)
        if extra not in chosen:
            chosen.append(extra)

    return chosen[:5]


def get_five_sentences_for_form(all_sentences: list) -> list:
    return pick_five_covering_alphabet(all_sentences)


def calculate_writing_box_dimensions():
    DPI = 300
    width_mm = (2000 / DPI) * 25.4
    height_mm = (200 / DPI) * 25.4
    return width_mm * mm, height_mm * mm


def draw_corner_marker(c, x, y, size=2.5 * mm):
    c.setFillColorRGB(0, 0, 0)
    c.circle(x, y, size, stroke=0, fill=1)


def draw_checkbox(c, x, y, size=3 * mm):
    c.rect(x, y, size, size, stroke=1, fill=0)


def create_form(form_id, sentences, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin_left = 15 * mm
    margin_right = width - 15 * mm
    box_width, box_height = calculate_writing_box_dimensions()
    box_x = (width - box_width) / 2

    y = height - 18 * mm
    c.setFont(FONT_BOLD, 14)
    title = "LIETUVIŠKO RANKRAŠČIO TYRIMAS"
    c.drawString((width - c.stringWidth(title, FONT_BOLD, 14)) / 2, y, title)

    y -= 7 * mm
    c.setFont(FONT_REGULAR, 8)
    desc = "Šis bakalauro darbo tyrimas skirtas sukurti dirbtinio intelekto sistemą,"
    c.drawString((width - c.stringWidth(desc, FONT_REGULAR, 8)) / 2, y, desc)

    y -= 3.5 * mm
    desc2 = "kuri atpažintų ranka rašytą lietuvišką tekstą."
    c.drawString((width - c.stringWidth(desc2, FONT_REGULAR, 8)) / 2, y, desc2)

    y -= 5 * mm
    c.setLineWidth(0.5)
    c.line(margin_left, y, margin_right, y)

    y -= 6 * mm
    c.setFont(FONT_BOLD, 9)
    c.drawString(margin_left, y, f"Forma Nr: {form_id:03d} (langelyje žymėti X)")

    y -= 5.5 * mm
    c.setFont(FONT_BOLD, 8)
    c.drawString(margin_left, y, "Lytis:")
    c.setFont(FONT_REGULAR, 8)
    checkbox_start = margin_left + 13 * mm
    draw_checkbox(c, checkbox_start, y - 0.8 * mm)
    c.drawString(checkbox_start + 4.5 * mm, y, "Vyras")
    draw_checkbox(c, checkbox_start + 18 * mm, y - 0.8 * mm)
    c.drawString(checkbox_start + 22.5 * mm, y, "Moteris")
    draw_checkbox(c, checkbox_start + 38 * mm, y - 0.8 * mm)
    c.drawString(checkbox_start + 42.5 * mm, y, "Kita")

    y -= 5.5 * mm
    c.setFont(FONT_BOLD, 8)
    c.drawString(margin_left, y, "Esu:")
    c.setFont(FONT_REGULAR, 8)
    draw_checkbox(c, checkbox_start, y - 0.8 * mm)
    c.drawString(checkbox_start + 4.5 * mm, y, "Dešiniarankis/ė")
    draw_checkbox(c, checkbox_start + 38 * mm, y - 0.8 * mm)
    c.drawString(checkbox_start + 42.5 * mm, y, "Kairiarankis/ė")

    y -= 5.5 * mm
    c.setFont(FONT_REGULAR, 7)
    c.drawString(margin_left, y, "El. paštas (jei norite sužinoti rezultatus):")
    y -= 3.5 * mm
    c.setLineWidth(0.3)
    c.line(margin_left, y, margin_right, y)
    y -= 4.5 * mm
    c.setLineWidth(0.5)
    c.line(margin_left, y, margin_right, y)

    y -= 6 * mm
    c.setFont(FONT_BOLD, 8)
    c.drawString(margin_left, y,
                 "UŽDUOTIS: Prašome TIKSLIAI ir AIŠKIAI perrašyti šiuos sakinius neužlendant už kraštų:")

    y -= 8 * mm
    for i, sentence in enumerate(sentences, 1):
        c.setFillColorRGB(0.95, 0.95, 0.95)
        sentence_box_height = 6 * mm
        c.rect(box_x, y - sentence_box_height, box_width, sentence_box_height, stroke=1, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont(FONT_REGULAR, 8)
        c.drawString(box_x + 3 * mm, y - sentence_box_height / 2 - 1 * mm, f"{i}. {sentence}")
        y -= sentence_box_height + 1.5 * mm
        c.setFillColorRGB(1, 1, 1)
        c.setStrokeColorRGB(0.3, 0.3, 0.3)
        c.setLineWidth(0.5)
        c.rect(box_x, y - box_height, box_width, box_height, stroke=1, fill=1)
        y -= box_height + 3 * mm

    footer_y = 18 * mm
    c.setFillColorRGB(0, 0, 0)
    c.setFont(FONT_BOLD, 9)
    thanks = "Ačiū už jūsų dalyvavimą tyrime!"
    c.drawString((width - c.stringWidth(thanks, FONT_BOLD, 9)) / 2, footer_y, thanks)

    marker_margin = 10 * mm
    marker_size = 2.5 * mm
    draw_corner_marker(c, marker_margin, height - marker_margin, marker_size)
    draw_corner_marker(c, width - marker_margin, height - marker_margin, marker_size)
    draw_corner_marker(c, marker_margin, marker_margin, marker_size)
    draw_corner_marker(c, width - marker_margin, marker_margin, marker_size)
    c.save()


def generate_forms_with_mapping(num_forms=500, output_dir='Forms_Real_Data'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pdfs", exist_ok=True)

    all_sentences = generate_lithuanian_sentences(num_sentences=num_forms * 10)
    forms_data = []

    for form_id in range(1, num_forms + 1):
        form_sentences = get_five_sentences_for_form(all_sentences)
        pdf_path = f"{output_dir}/pdfs/form_{form_id:03d}.pdf"
        create_form(form_id, form_sentences, pdf_path)
        forms_data.append({
            'form_id': form_id,
            'sentence_1': form_sentences[0],
            'sentence_2': form_sentences[1],
            'sentence_3': form_sentences[2],
            'sentence_4': form_sentences[3],
            'sentence_5': form_sentences[4]
        })
        if form_id % 50 == 0:
            print(f"Sukurta {form_id}/{num_forms} formu")

    df = pd.DataFrame(forms_data)
    mapping_path = f"{output_dir}/forms_mapping.csv"
    df.to_csv(mapping_path, index=False, encoding='utf-8')

    print(f"Baigta. Sukurtos {num_forms} formos.")
    print(f"PDF failai: {output_dir}/pdfs/")
    print(f"Mapping CSV: {mapping_path}")
    return mapping_path


if __name__ == "__main__":
    mapping_path = generate_forms_with_mapping(num_forms=500)
