# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class FormScanner:
    def __init__(self, forms_mapping_path, coordinates_path):
        self.forms_map = pd.read_csv(forms_mapping_path)
        with open(coordinates_path, 'r') as f:
            self.coords = json.load(f)

    def detect_corner_markers(self, image, debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = image.shape[:2]
        margin = 0.10

        corner_zones = {
            'top_left': {'x_min': 0, 'x_max': width * margin,
                         'y_min': 0, 'y_max': height * margin},
            'top_right': {'x_min': width * (1 - margin), 'x_max': width,
                          'y_min': 0, 'y_max': height * margin},
            'bottom_left': {'x_min': 0, 'x_max': width * margin,
                            'y_min': height * (1 - margin), 'y_max': height},
            'bottom_right': {'x_min': width * (1 - margin), 'x_max': width,
                             'y_min': height * (1 - margin), 'y_max': height},
        }

        def is_in_zone(cx, cy, zone):
            return (zone['x_min'] <= cx <= zone['x_max'] and
                    zone['y_min'] <= cy <= zone['y_max'])

        min_area, max_area = 100, 5000
        min_circ, max_circ = 0.5, 1.5

        all_candidates = []
        found_markers = {}

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if min_area < area < max_area and min_circ < circularity < max_circ:
                all_candidates.append({'cx': cx, 'cy': cy, 'area': area,
                                       'circ': circularity, 'in_zone': None})

            if not (min_circ < circularity < max_circ and min_area < area < max_area):
                continue

            corners_map = {
                'top_left': (0, 0),
                'top_right': (width, 0),
                'bottom_left': (0, height),
                'bottom_right': (width, height),
            }

            for zone_name, zone in corner_zones.items():
                if is_in_zone(cx, cy, zone):
                    if all_candidates:
                        all_candidates[-1]['in_zone'] = zone_name

                    if zone_name not in found_markers:
                        found_markers[zone_name] = (cx, cy, area)
                    else:
                        corner = corners_map[zone_name]
                        existing_cx, existing_cy, _ = found_markers[zone_name]
                        dist_new = np.sqrt((cx - corner[0]) ** 2 + (cy - corner[1]) ** 2)
                        dist_existing = np.sqrt((existing_cx - corner[0]) ** 2 +
                                                (existing_cy - corner[1]) ** 2)
                        if dist_new < dist_existing:
                            found_markers[zone_name] = (cx, cy, area)

        if debug:
            print(f"  Vaizdo dydis: {width}x{height}")
            print(f"  Rasta kandidatų: {len(all_candidates)}")
            for zone_name, (cx, cy, area) in found_markers.items():
                print(f"  Rastas: {zone_name}: ({cx}, {cy})")
            missing = set(corner_zones.keys()) - set(found_markers.keys())
            if missing:
                print(f"  Truksta: {', '.join(missing)}")

        if len(found_markers) != 4:
            self.save_debug_image(image, found_markers, all_candidates)
            return None

        return [
            (found_markers['top_left'][0], found_markers['top_left'][1]),
            (found_markers['top_right'][0], found_markers['top_right'][1]),
            (found_markers['bottom_left'][0], found_markers['bottom_left'][1]),
            (found_markers['bottom_right'][0], found_markers['bottom_right'][1]),
        ]

    def save_debug_image(self, image, found_markers, all_candidates,
                         save_path='debug_markers.png'):
        debug_img = image.copy()
        height, width = debug_img.shape[:2]
        margin = 0.10

        zones = [
            ((0, 0), (int(width * margin), int(height * margin))),
            ((int(width * (1 - margin)), 0), (width, int(height * margin))),
            ((0, int(height * (1 - margin))), (int(width * margin), height)),
            ((int(width * (1 - margin)), int(height * (1 - margin))), (width, height)),
        ]
        for (p1, p2) in zones:
            cv2.rectangle(debug_img, p1, p2, (255, 0, 0), 3)

        for cand in all_candidates:
            color = (0, 255, 255) if cand['in_zone'] else (0, 165, 255)
            cv2.circle(debug_img, (cand['cx'], cand['cy']), 15, color, 2)
            cv2.putText(debug_img, f"{int(cand['area'])}",
                        (cand['cx'] + 20, cand['cy']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for zone_name, (cx, cy, area) in found_markers.items():
            cv2.circle(debug_img, (cx, cy), 20, (0, 255, 0), -1)
            cv2.putText(debug_img, zone_name[:2].upper(), (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(debug_img, f"Found: {len(found_markers)}/4 markers",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(save_path, debug_img)
        print(f"  Debug vaizdas issaugotas: {save_path}")

    def manual_corner_selection(self, image):
        print("\nMANUAL MODE: Paspauskite 4 kampus sia tvarka:")
        print("  1. Virsus kaire")
        print("  2. Virsus desine")
        print("  3. Apacia kaire")
        print("  4. Apacia desine")

        corners = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append((x, y))
                cv2.circle(display, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(display, str(len(corners)), (x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Select Corners', display)

        scale = min(1.0, 1200 / image.shape[1], 800 / image.shape[0])
        display = cv2.resize(image, None, fx=scale, fy=scale)
        cv2.imshow('Select Corners', display)
        cv2.setMouseCallback('Select Corners', mouse_callback)

        while len(corners) < 4:
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                return None

        cv2.waitKey(500)
        cv2.destroyAllWindows()
        return [(int(x / scale), int(y / scale)) for (x, y) in corners]

    def perspective_transform(self, image, corners):
        actual_height, actual_width = image.shape[:2]
        (tl, tr, bl, br) = corners

        src = np.array([tl, tr, bl, br], dtype="float32")
        dst = np.array([
            [0, 0],
            [actual_width - 1, 0],
            [0, actual_height - 1],
            [actual_width - 1, actual_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (actual_width, actual_height))

    def extract_form_number_from_filename(self, scan_path):
        filename = Path(scan_path).stem
        match = re.search(r'(\d{4})', filename)
        if match:
            form_id = int(match.group(1))
            if 1 <= form_id <= 9999:
                return form_id
        return None

    def detect_checkboxes(self, corrected_image):
        if len(corrected_image.shape) == 3:
            gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = corrected_image.copy()

        TARGET_W, TARGET_H = 2480, 3508
        gray_norm = cv2.resize(gray, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_LINEAR)

        CHECKBOX_COORDS = {
            'vyras': (366, 554),
            'moteris': (562, 554),
            'kita': (784, 554),
            'desiniarankis': (366, 619),
            'kairiarankis': (784, 619),
        }

        INNER_BS, OUTER_BS = 13, 35
        scores = {}

        for name, (cx, cy) in CHECKBOX_COORDS.items():
            inner = gray_norm[cy - INNER_BS:cy + INNER_BS, cx - INNER_BS:cx + INNER_BS]
            outer = gray_norm[cy - OUTER_BS:cy + OUTER_BS, cx - OUTER_BS:cx + OUTER_BS]
            scores[name] = float(outer.mean() - inner.mean()) \
                if inner.size > 0 and outer.size > 0 else 0.0

        lytis_sorted = sorted(['vyras', 'moteris', 'kita'],
                              key=lambda k: scores[k], reverse=True)
        esu_sorted = sorted(['desiniarankis', 'kairiarankis'],
                            key=lambda k: scores[k], reverse=True)

        gender_conf = scores[lytis_sorted[0]] - scores[lytis_sorted[1]]
        hand_conf = scores[esu_sorted[0]] - scores[esu_sorted[1]]
        MIN_CONF = 0.5

        return {
            'gender': lytis_sorted[0] if gender_conf > MIN_CONF else None,
            'handedness': esu_sorted[0] if hand_conf > MIN_CONF else None,
            'gender_confidence': gender_conf,
            'handedness_confidence': hand_conf,
        }

    def extract_sentence_regions(self, corrected_image):
        regions = []
        actual_height, actual_width = corrected_image.shape[:2]
        expected_width = self.coords['page_dimensions']['width_px']
        expected_height = self.coords['page_dimensions']['height_px']
        scale_x = actual_width / expected_width
        scale_y = actual_height / expected_height

        for sent_info in self.coords['sentences']:
            x = max(0, int(sent_info['x'] * scale_x))
            y = max(0, int(sent_info['y'] * scale_y))
            w = int(sent_info['width'] * scale_x)
            h = int(sent_info['height'] * scale_y)
            x = min(x, actual_width - w)
            y = min(y, actual_height - h)
            regions.append(corrected_image[y:y + h, x:x + w])

        return regions

    def process_scan(self, scan_path, output_base_dir='Datasets/Real_Handwriting'):
        print(f"\n{'=' * 60}")
        print(f"Apdorojamas: {scan_path}")
        print(f"{'=' * 60}")

        image = cv2.imread(scan_path)
        if image is None:
            print("Nepavyko nuskaityti vaizdo")
            return None

        print("1. Aptinkami kampai...")
        corners = self.detect_corner_markers(image)
        if corners is None:
            print("Automatinis aptikimas nepavyko, pereinama i manual rezima...")
            corners = self.manual_corner_selection(image)
            if corners is None:
                print("Praleista")
                return None
        print("   Kampai rasti")

        print("2. Perspektyvos korekcija...")
        corrected = self.perspective_transform(image, corners)

        print("3. Formos numeris...")
        form_id = self.extract_form_number_from_filename(scan_path)
        if form_id is None:
            print(f"   Klaida: failo pavadinimas neatitinka formato (reikalingas 0001-9999)")
            return None
        print(f"   Forma Nr: {form_id:04d}")

        form_row = self.forms_map[self.forms_map['form_id'] == form_id]
        if form_row.empty:
            print(f"   Forma {form_id} nerasta mapping faile")
            return None

        sentences = [form_row.iloc[0][f'sentence_{i}'] for i in range(1, 6)]

        print("4. Checkbox aptikimas...")
        metadata = self.detect_checkboxes(corrected)
        print(f"   Lytis: {metadata['gender'] or 'nenustatyta'}")
        print(f"   Rankos: {metadata['handedness'] or 'nenustatyta'}")

        print("5. Eiluciu iskyrimas...")
        os.makedirs(f"{output_base_dir}/images", exist_ok=True)
        regions = self.extract_sentence_regions(corrected)
        annotations = []

        for i, (region, sentence) in enumerate(zip(regions, sentences), 1):
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            filepath = f"{output_base_dir}/images/form_{form_id:03d}_sentence_{i}.png"
            cv2.imwrite(filepath, region)
            annotations.append([filepath, sentence])

        print(f"   Issaugota {len(annotations)} sakiniu")
        return annotations, metadata


def batch_process_scans(scans_folder, forms_mapping_path, coordinates_path,
                        output_dir='Datasets/Real_Handwriting'):
    print(f"\n{'=' * 60}")
    print("BATCH SKENAVIMO APDOROJIMAS")
    print(f"{'=' * 60}\n")

    scanner = FormScanner(forms_mapping_path, coordinates_path)

    scan_files = sorted(set(
        f for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG']
        for f in Path(scans_folder).glob(ext)
    ))

    if not scan_files:
        print(f"Nerasta skenavimų: {scans_folder}")
        return

    print(f"Rasta {len(scan_files)} skenavimų\n")

    all_annotations = []
    stats = {
        'gender': {'vyras': 0, 'moteris': 0, 'kita': 0, 'nenustatyta': 0},
        'handedness': {'desiniarankis': 0, 'kairiarankis': 0, 'nenustatyta': 0},
    }

    for scan_file in scan_files:
        result = scanner.process_scan(str(scan_file), output_dir)
        if not result:
            continue
        annotations, metadata = result
        all_annotations.extend(annotations)

        g = metadata.get('gender') or 'nenustatyta'
        h = metadata.get('handedness') or 'nenustatyta'
        stats['gender'][g] = stats['gender'].get(g, 0) + 1
        stats['handedness'][h] = stats['handedness'].get(h, 0) + 1

    if not all_annotations:
        print("Nepavyko apdoroti nei vieno skenavimo")
        return

    csv_path = f"{output_dir}/annotations.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        for image_path, label in all_annotations:
            f.write(f"{image_path},{label}\n")

    print(f"\n{'=' * 60}")
    print(f"BAIGTA")
    print(f"{'=' * 60}")
    print(f"Formu:   {len(all_annotations) // 5}")
    print(f"Sakiniu: {len(all_annotations)}")
    print(f"\nLytis:")
    for k, v in stats['gender'].items():
        if v: print(f"  {k}: {v}")
    print(f"\nRankos:")
    for k, v in stats['handedness'].items():
        if v: print(f"  {k}: {v}")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    batch_process_scans(
        scans_folder='Scans',
        forms_mapping_path='Forms_Real_Data/forms_mapping.csv',
        coordinates_path='Forms_Real_Data/extraction_coordinates.json'
    )
