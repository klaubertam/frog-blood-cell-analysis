"""Predict masks, propose nucleus-seeded watershed instances, save one cell bag per photo."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed, mark_boundaries
import torch
from segmentation import (fresh, dump, csvwrite, load_image, make_model, model_input,
                          restore, encode, ignored_region, IGNORE, EXTS, photo_id)


def separate(y, minimum_nucleus_area=3):
    """Operate on the resized model grid; no biological shape rejection."""
    cells = np.isin(y, [1, 2])
    nuclei, _ = ndi.label(y == 2)
    sizes = np.bincount(nuclei.ravel())
    keep = sizes >= minimum_nucleus_area
    keep[0] = False
    markers, _ = ndi.label(keep[nuclei])
    components, _ = ndi.label(cells)
    seeded = np.unique(components[markers > 0])
    seeded = seeded[seeded > 0]
    domain = np.isin(components, seeded)
    instances = watershed(-ndi.distance_transform_edt(domain), markers, mask=domain).astype(np.int32)
    return instances, cells & ~domain


def square_crop(rgb, region, size=64):
    yy, xx = np.where(region)
    crop = rgb[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
    crop[~region[yy.min():yy.max()+1, xx.min():xx.max()+1]] = 0
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = crop
    return np.array(Image.fromarray(canvas).resize((size, size), Image.Resampling.BILINEAR))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan', help='Frozen plan.json for the experiment')
    source.add_argument('--new-images', help='Folder of unseen Foto NUMBER images, with no labels needed')
    p.add_argument('--model', required=True, help='Segmentation checkpoint trained with this plan')
    p.add_argument('--out', default='C_bags')
    p.add_argument('--min-nucleus-area', type=int, default=3, help='Pixels on resized model grid; tune on training/validation only')
    p.add_argument('--min-cell-area', type=int, default=20, help='Pixels on resized model grid; technical noise filter')
    p.add_argument('--pink-adjacent', choices=['exclude','keep'], default='exclude',
                   help='Whether to exclude or retain cells adjacent to predicted magenta structures; examine both policies')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if min(a.min_nucleus_area, a.min_cell_area, a.threads) < 1:
        p.error('Area thresholds and threads must be positive')
    torch.set_num_threads(a.threads)
    state = torch.load(a.model, map_location='cpu', weights_only=True)
    if a.plan:
        plan = json.loads(Path(a.plan).read_text())
    else:
        records = []
        for file in sorted(Path(a.new_images).rglob('*')):
            if file.suffix.lower() not in EXTS or 'mask' in file.stem.lower(): continue
            rgb = load_image(file)
            records.append({'id':photo_id(file),'image':str(file.resolve()),'target':None,
                'pixel_sha256':hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()})
        ids = [r['id'] for r in records]
        if not ids or len(set(ids)) != len(ids): raise ValueError('No images or duplicate Foto IDs')
        plan = {'records':records,'splits':{'inference':ids}}
    if a.plan and state.get('splits') != plan['splits']:
        raise ValueError('Checkpoint has a different/missing split. Retrain segmentation with this plan to protect held-out images.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(state['base']).to(device)
    model.load_state_dict(state['model']); model.eval()
    out = fresh(a.out)
    for name in ['bags', 'masks', 'instances', 'previews']:
        (out/name).mkdir()
    bag_records, features, quality = [], [], []
    with torch.no_grad():
        for record in plan['records']:
            key = record['id']; rgb = load_image(record['image'])
            digest = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
            if digest != record['pixel_sha256']:
                raise ValueError(f'Image {key} changed after the split was frozen')
            x, valid_grid, meta = model_input(rgb, np.zeros(rgb.shape[:2], np.uint8), state['size'], state['ignore_box'])
            tensor = torch.from_numpy(x.transpose(2,0,1).copy()).float()[None].to(device)/255
            y = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
            y[valid_grid == IGNORE] = IGNORE
            instances, unassigned = separate(y, a.min_nucleus_area)
            bad_border = ndi.binary_dilation(y == IGNORE)
            bad_border[[0,-1],:] = True; bad_border[:,[0,-1]] = True
            accepted = np.zeros_like(instances); crops = []; kept_ids = []
            for region in regionprops(instances):
                cell = instances == region.label
                nucleus = cell & (y == 2)
                flags = []
                if region.area < a.min_cell_area: flags.append('small')
                if np.any(cell & bad_border): flags.append('border_or_ignored_region')
                adjacent_pink=bool(np.any(ndi.binary_dilation(cell) & (y == 3)))
                if adjacent_pink and a.pink_adjacent == 'exclude': flags.append('adjacent_pink_region')
                # Shape is measured, not used to remove potentially abnormal cells.
                quality.append({'id': key, 'instance_id': region.label,
                                'adjacent_pink_region':int(adjacent_pink),
                                'status':';'.join(flags) if flags else ('candidate_kept_pink_adjacent_REVIEW' if adjacent_pink else 'candidate_kept')})
                if flags: continue
                accepted[cell] = region.label
                kept_ids.append(region.label); crops.append(square_crop(x, cell))
                nr = regionprops(nucleus.astype(np.uint8))[0]
                L, W = region.axis_major_length, region.axis_minor_length
                NL, NW = nr.axis_major_length, nr.axis_minor_length
                ellipse = np.pi*L*W/4
                features.append({'id':key,'instance_id':region.label,
                    'cell_area_grid_px':int(region.area), 'nucleus_area_grid_px':int(nucleus.sum()),
                    'cell_major_grid_px':L,'cell_minor_grid_px':W,
                    'nucleus_major_grid_px':NL,'nucleus_minor_grid_px':NW,
                    'cell_axis_ratio':L/W if W else None,'nucleus_axis_ratio':NL/NW if NW else None,
                    'nuclear_to_cell_area':float(nucleus.sum()/region.area),
                    'nuclear_to_cell_ellipse_area':(np.pi*NL*NW/4)/ellipse if ellipse else None,
                    'cell_solidity':region.solidity,'cell_eccentricity':region.eccentricity})
            bag = np.stack(crops) if crops else np.empty((0,64,64,3), np.uint8)
            bag_file = f'bags/{key}.npz'
            np.savez_compressed(out/bag_file, crops=bag, instance_ids=np.array(kept_ids, dtype=np.int32))
            role = next(role for role, ids in plan['splits'].items() if key in ids)
            bag_records.append({'id':key,'target':record['target'],'split':role,'file':bag_file,
                'instances':len(crops),'unassigned_cell_pixels':int(unassigned.sum()),
                'status':'usable' if crops else 'ABSTAIN: no retained cells'})
            Image.fromarray(encode(y)).save(out/'masks'/f'{key}_model_grid.png')
            original = restore(y, meta)
            original[ignored_region(original.shape, state['ignore_box'])] = IGNORE
            Image.fromarray(encode(original)).save(out/'masks'/f'{key}_original_size.png')
            Image.fromarray(instances).save(out/'instances'/f'{key}_all_candidates.tif')
            Image.fromarray(accepted).save(out/'instances'/f'{key}_retained.tif')
            Image.fromarray(unassigned.astype(np.uint8)*255).save(out/'instances'/f'{key}_unassigned.png')
            overlay = (mark_boundaries(x/255., accepted)*255).astype(np.uint8)
            Image.fromarray(overlay).save(out/'previews'/f'{key}.jpg')
            print(key, 'retained candidates:', len(crops), flush=True)
    dump(out/'bags.json', {'records':bag_records,'splits':plan['splits'],
        'classification_labels':{'0':'normal','1':'anomaly'},
        'segmentation_model_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        'settings':vars(a),'measurement_grid':state['size']})
    csvwrite(out/'bag_quality.csv', bag_records)
    csvwrite(out/'cell_features.csv', features)
    csvwrite(out/'instance_quality.csv', quality)
    print('Review previews. Watershed instances are proposals, not verified counts or cell diagnoses.')

if __name__ == '__main__':
    main()
