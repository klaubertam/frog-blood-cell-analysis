"""Experiment 2: binary whole-cell U-Net, image-level evaluation only.
Run --check-only first. See README.md for the exact experimental protocol.
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from preprocessing import (find_pairs, read_rgb, read_mask, letterbox, pixel_hash,
                           save_preview, restore_probability, index_folder,
                           hole_statistics, ignore_coverage, check_ignore_box_fraction)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_splits(n, folds, val_fraction, seed):
    """Each image is in the outer test set once. Inner validation selects epoch."""
    if not 2 <= folds <= n or not 0 < val_fraction < .5:
        raise ValueError('Need 2 <= folds <= number of pairs and 0 < val-fraction < 0.5')
    shuffled = np.random.default_rng(seed).permutation(n)
    result = []
    for fold, test in enumerate(np.array_split(shuffled, folds), 1):
        remaining = np.array(sorted(set(range(n)) - set(test.tolist())))
        remaining = np.random.default_rng(seed + fold).permutation(remaining)
        nv = max(1, round(len(remaining) * val_fraction))
        val, train = remaining[:nv], remaining[nv:]
        if len(train) < 2:
            raise ValueError('Too few training images after splitting')
        result.append({'fold': fold, 'train': train.tolist(), 'validation': val.tolist(), 'test': test.tolist()})
    return result


def audit(pairs, out, size, crop_box, ignore_box_frac=None, ignore_masks=None, clahe=False):
    prepared, rows, seen = [], [], {}
    previews = out / 'preprocessing_previews'
    previews.mkdir()
    ignore_paths = index_folder(ignore_masks) if ignore_masks is not None else {}
    known = {entry[2] for entry in pairs}
    extra = sorted(set(ignore_paths) - known)
    if extra:
        raise ValueError(f'Ignore-mask IDs do not match annotated photos: {extra}')
    for ip, mp, key in pairs:
        rgb, mask = read_rgb(ip), read_mask(mp)
        digest = pixel_hash(rgb)
        if digest in seen:
            raise ValueError(f'Duplicate photograph pixels: IDs {seen[digest]} and {key}. Remove duplicate before splitting.')
        seen[digest] = key
        imask = read_mask(ignore_paths[key]) if key in ignore_paths else None
        orig_ignored = ignore_coverage(rgb.shape, ignore_box_frac, imask)
        hole_count, hole_area = hole_statistics(mask)
        if hole_count:
            print(f'REVIEW ID {key}: {hole_count} enclosed black hole(s), '
                  f'{hole_area} pixels. Are any nuclei incorrectly left black?')
        if np.any(orig_ignored & (mask != 0)):
            print(f'REVIEW ID {key}: ignore regions cover '
                  f'{int(np.sum(orig_ignored & (mask != 0)))} labeled-cell pixels.')
        image, target, valid, meta = letterbox(
            rgb, mask, size, crop_box, ignore_box_frac,
            ignore_mask=imask, clahe=clahe)
        meta['original_ignored'] = orig_ignored
        prepared.append((image, target, valid, meta))
        l, t, r, b = meta['crop_box']
        fg = float(mask[t:b, l:r].mean())
        if valid.sum() == 0:
            raise ValueError(f'No valid pixels remaining for {key}; narrow the exclusions.')
        processed_fg = float(target.sum()/valid.sum())
        if fg in (0, 1):
            print(f'CHECK ID {key}: foreground fraction is {fg:.0%}; confirm the annotation.')
        if fg > 0 and target.sum() == 0:
            raise ValueError(f'All foreground disappeared during resize for ID {key}. Increase --size.')
        rows.append({'image_id': key, 'image_path': str(ip), 'mask_path': str(mp),
                     'image_pixel_sha256': digest, 'mask_file_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
                     'original_width': rgb.shape[1], 'original_height': rgb.shape[0],
                     'crop_box': json.dumps(meta['crop_box']), 'resized_width': meta['width'],
                     'resized_height': meta['height'], 'foreground_fraction_before': fg,
                     'foreground_fraction_after': processed_fg,
                     'enclosed_hole_count_review': hole_count,
                     'enclosed_hole_area_pixels_review': hole_area,
                     'ignored_fraction_original': float(orig_ignored.mean()),
                     'valid_fraction_model_grid': float(valid.mean())})
        save_preview(previews / f'{key}.png', image, target, valid, key)
    write_csv(out/'data_audit.csv', rows)
    return prepared


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', default='experiment_2_results')
    ap.add_argument('--expected-pairs', type=int, default=60)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--crop-box', type=int, nargs=4, metavar=('LEFT','TOP','RIGHT','BOTTOM'),
                    help='Optional verified common crop in original pixels; default keeps the full image.')
    ap.add_argument('--timestamp-box-frac', type=float, nargs=4,
                    metavar=('LEFT','TOP','RIGHT','BOTTOM'),
                    help='Optional confirmed fractional timestamp rectangle; excludes only this area, not an entire image strip.')
    ap.add_argument('--ignore-masks',
                    help='Optional folder with binary masks: white = pixels to ignore (e.g. uncertain magenta regions).')
    ap.add_argument('--augmentation', choices=['conservative', 'artis', 'none'],
                    default='conservative',
                    help='Training-only paired transformations. ARTIS mode adds ±20° rotation, ±5%% shifts, ±5° shear, ±8%% zoom.')
    ap.add_argument('--clahe', action='store_true',
                    help='Optional LAB-luminance CLAHE; applied identically to training and inference.')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--val-fraction', type=float, default=.125)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--train-final', action='store_true', help='After CV, fit all pairs for the median selected epoch count.')
    args = ap.parse_args()
    check_ignore_box_fraction(args.timestamp_box_frac)
    if args.size < 32 or args.size % 16:
        ap.error('--size must be >=32 and divisible by 16')
    if min(args.epochs, args.patience, args.batch_size, args.base, args.threads) < 1 or args.lr <= 0:
        ap.error('epochs, patience, batch-size, base, threads and lr must be positive')
    return args


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty. Choose a new output folder; existing experiments are never overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    pairs, unlabeled = find_pairs(args.images, args.masks, args.expected_pairs)
    splits = make_splits(len(pairs), args.folds, args.val_fraction, args.seed)
    config = {'experiment': 'Experiment 2', 'task': 'binary whole-cell segmentation',
              'label_map': {'background': 0, 'whole_cell_including_nuclear_pixels': 1},
              'normalization': 'RGB / 255', 'resize': 'aspect-preserving letterbox',
              'timestamp_policy': 'excluded from input, loss, and metrics when box supplied',
              'uncertain_region_policy': 'optional white=ignore masks; otherwise supervised background',
              'color_enhancement': 'LAB CLAHE' if args.clahe else 'none',
              'augmentation': args.augmentation,
              'threshold': .5, 'evaluation_grid': 'resized real-image region; padding ignored',
              'split_unit': 'photograph, no animal grouping', **vars(args)}
    write_json(out/'config.json', config)
    write_json(out/'unlabeled_image_ids.json', unlabeled)
    id_splits = [{'fold': s['fold'], **{role: [pairs[i][2] for i in s[role]] for role in ('train','validation','test')}} for s in splits]
    write_json(out/'splits.json', id_splits)
    source = out/'source_code'
    source.mkdir()
    for name in ('train_unet.py','preprocessing.py','augmentation.py','model.py','predict.py','requirements.txt','README.md'):
        shutil.copy2(Path(__file__).resolve().parent/name, source/name)
    prepared = audit(pairs, out, args.size, args.crop_box,
                     args.timestamp_box_frac, args.ignore_masks, args.clahe)
    print(f'Validated {len(pairs)} pairs; {len(unlabeled)} other photographs have no masks and are excluded from training.')
    print('Inspect preprocessing_previews before training. Alignment and mask completeness require visual review.')
    if args.check_only:
        write_json(out/'status.json', {'status': 'audit_only_no_model_trained'})
        return

    import torch
    from torch.utils.data import Dataset, DataLoader
    from model import UNet, masked_loss, overlap_scores
    from augmentation import augment_pair
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install a suitable PyTorch build or use --device cpu.')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    write_json(out/'environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
                                      'numpy': np.__version__, 'device': device, 'platform': platform.platform()})
    write_json(out/'status.json', {'status': 'training'})

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    class Cells(Dataset):
        def __init__(self, indices, augment=False):
            self.indices, self.augment = indices, augment
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, index):
            idx = self.indices[index]
            rgb, mask, valid, _ = prepared[idx]
            if self.augment:
                rgb, mask, valid = augment_pair(rgb, mask, valid, args.augmentation)
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255
            y = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
            v = torch.from_numpy(np.ascontiguousarray(valid[None])).float()
            return x, y, v, idx

    def loader(indices, training=False):
        return DataLoader(Cells(indices, training), batch_size=args.batch_size, shuffle=training,
                          num_workers=0, pin_memory=device == 'cuda')

    def epoch_train(model, data, optimizer):
        model.train()
        total, count = 0., 0
        for x, y, valid, _ in data:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y, valid)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; check inputs and learning rate')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x)
            count += len(x)
        return total/count

    @torch.no_grad()
    def evaluate(model, data, prediction_dir=None):
        model.eval()
        rows = []
        if prediction_dir:
            prediction_dir.mkdir()
        for x, y, valid, indices in data:
            probabilities = torch.sigmoid(model(x.to(device))).cpu()
            predicted = (probabilities >= .5).float()
            dice, iou = overlap_scores(predicted, y, valid)
            for j, idx_tensor in enumerate(indices):
                idx = int(idx_tensor)
                key = pairs[idx][2]
                rows.append({'image_id': key, 'dice': float(dice[j]), 'iou': float(iou[j]),
                             'target_pixels': int((y[j]*valid[j]).sum()),
                             'predicted_pixels': int((predicted[j]*valid[j]).sum())})
                if prediction_dir:
                    rgb, _, v, meta = prepared[idx]
                    binary = (predicted[j,0].numpy()*v).astype(np.uint8)
                    Image.fromarray(binary*255).save(prediction_dir/f'{key}_mask_model_grid.png')
                    restored = restore_probability(probabilities[j,0].numpy(), meta)
                    restored[meta['original_ignored']] = 0
                    Image.fromarray((restored >= .5).astype(np.uint8)*255).save(prediction_dir/f'{key}_mask_original_size.png')
                    save_preview(prediction_dir/f'{key}_prediction_overlay.png', rgb, binary, v, key)
        return rows

    def checkpoint(model, epoch, selected_on):
        return {'model': model.state_dict(), 'base': args.base, 'size': args.size,
                'crop_box': args.crop_box, 'timestamp_box_frac': args.timestamp_box_frac,
                'ignore_masks_used': args.ignore_masks is not None,
                'clahe': args.clahe, 'threshold': .5, 'epoch': epoch,
                'experiment': 'Experiment 2', 'selected_on': selected_on,
                'preprocessing_version': 2, 'normalization': 'RGB / 255'}

    all_rows, fold_rows, best_epochs = [], [], []
    for split in splits:
        fold = split['fold']
        seed_all(args.seed + fold)
        folder = out/f'fold_{fold}'
        folder.mkdir()
        train, val, test = [loader(split[r], r=='train') for r in ('train','validation','test')]
        print(f'Fold {fold}: train={len(split["train"])} validation={len(split["validation"])} test={len(split["test"])}; device={device}')
        model = UNet(args.base).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history, best, best_epoch, stale = [], -1., 0, 0
        for epoch in range(1, args.epochs+1):
            loss = epoch_train(model, train, optimizer)
            vr = evaluate(model, val)
            vd, vi = [float(np.mean([r[k] for r in vr])) for k in ('dice','iou')]
            history.append({'epoch': epoch, 'train_loss': loss, 'validation_dice': vd, 'validation_iou': vi})
            write_csv(folder/'history.csv', history)
            if vd > best:
                best, best_epoch, stale = vd, epoch, 0
                torch.save(checkpoint(model, epoch, 'inner validation Dice'), folder/'best_model.pt')
            else:
                stale += 1
            print(f'fold={fold} epoch={epoch} loss={loss:.4f} val_dice={vd:.4f} val_iou={vi:.4f}', flush=True)
            if stale >= args.patience:
                break
        state = torch.load(folder/'best_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        # Test data are only evaluated after checkpoint selection has finished.
        tr = evaluate(model, test, folder/'test_predictions')
        for row in tr:
            row['fold'] = fold
        write_csv(folder/'test_metrics.csv', tr)
        all_rows.extend(tr)
        best_epochs.append(best_epoch)
        fold_rows.append({'fold': fold, 'train_count': len(split['train']), 'validation_count': len(split['validation']),
                          'test_count': len(split['test']), 'best_epoch': best_epoch, 'best_validation_dice': best,
                          'test_dice': float(np.mean([r['dice'] for r in tr])), 'test_iou': float(np.mean([r['iou'] for r in tr]))})
        write_csv(out/'fold_metrics.csv', fold_rows)
        write_csv(out/'per_image_test_metrics.csv', all_rows)
    summary = {'completed_folds': args.folds, 'evaluated_images': len(all_rows),
               'mean_test_dice': float(np.mean([r['dice'] for r in all_rows])),
               'mean_test_iou': float(np.mean([r['iou'] for r in all_rows])),
               'sd_fold_test_dice': float(np.std([r['test_dice'] for r in fold_rows], ddof=1)),
               'evaluation': 'image-level outer cross-validation; inner validation selects checkpoint',
               'grid': 'letterboxed size, real-image pixels only; no original-resolution metric reported'}
    write_json(out/'summary.json', summary)
    if args.train_final:
        seed_all(args.seed + 10000)
        final_epochs = max(1, int(np.median(best_epochs)))
        final_model = UNet(args.base).to(device)
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
        final_loader = loader(list(range(len(pairs))), True)
        final_history = []
        for epoch in range(1, final_epochs+1):
            loss = epoch_train(final_model, final_loader, final_optimizer)
            final_history.append({'epoch': epoch, 'train_loss': loss})
            print(f'final epoch={epoch}/{final_epochs} loss={loss:.4f}', flush=True)
        torch.save(checkpoint(final_model, final_epochs, 'fixed median inner-selected epoch count; trained on all pairs'), out/'final_model.pt')
        write_csv(out/'final_training_history.csv', final_history)
    write_json(out/'status.json', {'status': 'completed'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
