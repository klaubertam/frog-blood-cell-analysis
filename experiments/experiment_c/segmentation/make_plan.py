"""Read image labels from the supplied workbook; freeze one image-level split."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import openpyxl
from sklearn.model_selection import train_test_split
from segmentation import fresh, dump, csvwrite, load_image


def read_targets(path):
    targets = {}
    audit = []
    for sheet in openpyxl.load_workbook(path, data_only=True, read_only=True):
        for row_number, row in enumerate(sheet.values, 1):
            if not row or row[0] is None:
                continue
            match = re.fullmatch(r'\s*foto\s*[_-]?\s*(\d+)\s*', str(row[0]), re.I)
            if not match:
                continue  # Title/header, not a photograph record.
            key = str(int(match.group(1)))
            value = row[1] if len(row) > 1 else None
            if key in targets or any(r['id'] == key for r in audit):
                raise ValueError(f'Duplicate workbook ID {key}')
            valid = str(value).strip() in ('0', '1', '0.0', '1.0')
            audit.append({'id': key, 'sheet': sheet.title, 'row': row_number,
                          'raw_label': value, 'status': 'valid' if valid else 'EXCLUDED: missing/invalid label'})
            if valid:
                targets[key] = int(float(value))
    if not targets:
        raise ValueError('No Foto NUMBER rows with labels 0 or 1 found')
    return targets, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Reviewed/prepared segmentation folder with pairs.json')
    p.add_argument('--targets', required=True, help='Original .xlsx; first column Foto ID, second column label')
    p.add_argument('--out', default='C_plan')
    p.add_argument('--val-count', type=int, default=4)
    p.add_argument('--test-count', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.val_count, a.test_count) < 2:
        p.error('Validation and test each need at least two images for both classes')
    targets, audit = read_targets(a.targets)
    data = Path(a.data).resolve()
    pairs = json.loads((data/'pairs.json').read_text())
    records = []
    hashes = {}
    for pair in pairs:
        key = pair['id']
        if key not in targets:
            raise ValueError(f'Photo {key} has no valid workbook label; resolve this before making the plan')
        image_path = data/pair['image']
        image = load_image(image_path)
        digest = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f'Duplicate image pixels: {key} and {hashes[digest]}')
        hashes[digest] = key
        records.append({'id': key, 'image': str(image_path), 'target': targets[key],
                        'pixel_sha256': digest})
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prepared image IDs')
    train_val, test = train_test_split(ids, test_size=a.test_count, random_state=a.seed,
                                       stratify=[targets[k] for k in ids])
    train, val = train_test_split(train_val, test_size=a.val_count, random_state=a.seed,
                                  stratify=[targets[k] for k in train_val])
    splits = dict(train=train, validation=val, test=test)
    if any({targets[k] for k in keys} != {0, 1} for keys in splits.values()):
        raise ValueError('Each split must contain normal AND anomaly; increase split sizes')
    out = fresh(a.out)
    dump(out/'splits.json', splits)
    dump(out/'plan.json', {'records': records, 'splits': splits, 'seed': a.seed,
                           'classification_labels': {'0': 'normal', '1': 'anomaly'}})
    csvwrite(out/'workbook_audit.csv', audit)
    csvwrite(out/'matched_targets.csv', [{'id':r['id'],'target':r['target'],
        'meaning':'anomaly' if r['target'] else 'normal',
        'split':next(s for s, keys in splits.items() if r['id'] in keys)} for r in records])
    counts = {s: {'originals': len(keys), 'normal': sum(targets[k] == 0 for k in keys),
                  'anomaly': sum(targets[k] == 1 for k in keys)} for s, keys in splits.items()}
    dump(out/'counts.json', counts)
    print(json.dumps(counts, indent=2))
    print('Four rotations give', 4*len(train), 'training views, not independent images.')

if __name__ == '__main__':
    main()
