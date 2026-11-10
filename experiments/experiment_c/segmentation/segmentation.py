"""Experiment C: prepare, train and predict four-class cell segmentation.
See START_HERE.md. Original masks require review after candidate conversion.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

NAMES = ['background', 'cytoplasm', 'nucleus', 'pink_region']
GRAY = np.array([0, 85, 170, 255], dtype=np.uint8)
COLORS = np.array([[0,0,0], [60,190,110], [50,120,255], [240,65,180]], dtype=np.uint8)
IGNORE = 255  # Internal ignored label. Saved editable masks use gray 200 for ignore.
EXTS = {'.jpg','.jpeg','.png','.tif','.tiff'}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def csvwrite(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fresh(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output folder is not empty: {path}. Use a new name to preserve previous work.')
    path.mkdir(parents=True,exist_ok=True)
    return path


def photo_id(path):
    m=re.search(r'foto[_\s-]*(\d+)',Path(path).stem,re.I)
    if not m: raise ValueError(f'Expected Foto NUMBER in filename: {path}')
    return str(int(m.group(1)))


def load_image(path, mask=False):
    with Image.open(path) as im:
        if getattr(im,'n_frames',1)!=1 or im.getexif().get(274,1)!=1:
            raise ValueError(f'Export a single, correctly oriented image first: {path}')
        if im.mode not in ('L','RGB'):
            raise ValueError(f'Expected 8-bit L or RGB image, got {im.mode}: {path}')
        a=np.array(im)
        if mask:
            if a.ndim==3:
                if not (np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])):
                    raise ValueError(f'Mask must be grayscale: {path}')
                a=a[:,:,0]
            return a
        return np.array(im.convert('RGB'))


def normalize_old_mask(a):
    """Explicit observed palette aliases. Near-black/white edges in Foto 24 only.
    Never choose a new semantic class from an arbitrary threshold.
    """
    known = (a<=24)|(a>=231)|np.isin(a,[64,85,170,192])
    if not known.all():
        raise ValueError(f'Unrecognized original mask values: {np.unique(a[~known]).tolist()}')
    out=np.zeros_like(a)
    out[a>=231]=255
    out[np.isin(a,[64,85])]=85
    out[np.isin(a,[170,192])]=170
    return out, int(np.count_nonzero(~np.isin(a,[0,64,85,170,192,255])))


def candidate_labels(a):
    """Heuristic drafts only. Hole filling cannot identify biology with certainty."""
    q, changed=normalize_old_mask(a)
    cyto=np.isin(q,[170,255])
    # Filling ONLY the cytoplasm boundary avoids filling holes inside free artifacts.
    enclosed=ndi.binary_fill_holes(cyto) & ~cyto
    y=np.zeros(q.shape,np.uint8)
    y[cyto]=1
    y[q==85]=3
    y[enclosed]=2
    count=int(ndi.label(enclosed)[1])
    return y, {'edge_pixels_normalized':changed, 'candidate_nucleus_components':count,
               'dark_pixels_reassigned_to_nucleus':int(np.count_nonzero(enclosed & (q==85)))}


def encode(y):
    saved=np.full(y.shape,200,np.uint8)
    for k,v in enumerate(GRAY):saved[y==k]=v
    return saved


def decode(a):
    valid=np.isin(a,[0,85,170,255,200])
    if not valid.all():raise ValueError(f'Edited labels contain unexpected values: {np.unique(a[~valid]).tolist()}. Use exact palette.')
    y=np.full(a.shape,IGNORE,np.uint8)
    for k,v in enumerate(GRAY):y[a==v]=k
    return y


def ignored_region(shape, box):
    v=np.zeros(shape,bool)
    if box is not None:
        l,t,r,b=box;h,w=shape
        v[int(t*h):int(math.ceil(b*h)),int(l*w):int(math.ceil(r*w))]=True
    return v


def colorize(y):
    c=np.full((*y.shape,3),100,np.uint8)
    for k in range(4):c[y==k]=COLORS[k]
    return c


def preview(path,rgb,y,title):
    h,w=y.shape; width=480;height=max(1,round(h*width/w))
    overlay=rgb.astype(float)
    use=(y>0)&(y<4)
    colors=colorize(y)
    overlay[use]=.6*overlay[use]+.4*colors[use]
    overlay[y==IGNORE]=[100,100,100]
    canvas=Image.new('RGB',(width*3,height+52),'white')
    for j,a in enumerate((rgb,colors,overlay.astype(np.uint8))):
        canvas.paste(Image.fromarray(a).resize((width,height),Image.Resampling.NEAREST if j==1 else Image.Resampling.BILINEAR),(j*width,52))
    d=ImageDraw.Draw(canvas);d.text((8,5),title,fill='black')
    d.text((8,25),'Photo | Labels: green=cytoplasm, blue=nucleus, pink=pink region, gray=ignored | Overlay',fill='black')
    canvas.save(path)


def prepare(args):
    out=fresh(args.out); source=out/'originals';source.mkdir()
    images={}; masks={}
    with zipfile.ZipFile(args.zip) as z:
        for info in z.infolist():
            p=Path(info.filename)
            if info.is_dir() or p.suffix.lower() not in EXTS or '__MACOSX' in p.parts:continue
            if info.file_size>100_000_000:raise ValueError('Individual image exceeds 100 MB limit')
            key=photo_id(p);is_mask='mask' in p.stem.lower(); dest=masks if is_mask else images
            if key in dest:raise ValueError(f'Duplicate {"mask" if is_mask else "image"} ID {key}')
            target=source/p.name
            if target.exists():raise ValueError(f'Duplicate basename {p.name}')
            target.write_bytes(z.read(info));dest[key]=target
    missing=sorted(set(images)-set(masks),key=int)
    orphan=sorted(set(masks)-set(images),key=int)
    if orphan:raise ValueError(f'Masks without photographs: {orphan}')
    keys=sorted(set(images)&set(masks),key=int)
    if not keys:raise ValueError('No image/mask pairs')
    for name in ['images','labels','review_previews']:(out/name).mkdir()
    records=[];pairs=[];hashes={}
    for key in keys:
        rgb=load_image(images[key]);a=load_image(masks[key],True)
        if rgb.shape[:2]!=a.shape:raise ValueError(f'Geometry mismatch for {key}')
        sha=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
        if sha in hashes:raise ValueError(f'Duplicate photograph pixels: {key}, {hashes[sha]}')
        hashes[sha]=key
        y,stats=candidate_labels(a)
        y[ignored_region(a.shape,args.ignore_box)]=IGNORE
        Image.fromarray(encode(y)).save(out/'labels'/f'{key}.png')
        photo_name=key+images[key].suffix.lower();shutil.copy2(images[key],out/'images'/photo_name)
        preview(out/'review_previews'/f'{key}.jpg',rgb,y,f'Foto {key}: CANDIDATE labels -- review against photograph')
        pairs.append({'id':key,'image':'images/'+photo_name,'label':f'labels/{key}.png'})
        records.append({'id':key,'width':a.shape[1],'height':a.shape[0],
                        'original_gray_values':' '.join(map(str,np.unique(a).tolist())),**stats,
                        'candidate_nucleus_pixels':int((y==2).sum()),'candidate_pink_pixels':int((y==3).sum()),
                        'review_status':'REQUIRED: holes, border nuclei, gray objects, missing structures'})
    dump(out/'pairs.json',pairs);dump(out/'preparation.json',{'ignore_box':args.ignore_box,'missing_masks':missing,
         'classes':NAMES,'saved_gray_values':GRAY.tolist(),'ignore_gray':200,'candidate_labels_require_review':True})
    csvwrite(out/'mask_audit.csv',records)
    print(f'Prepared {len(keys)} pairs. Photographs without masks: {missing}.')
    print(f'Review {out}/review_previews and correct {out}/labels before training.')


def model_input(rgb,y,size,box):
    rgb=rgb.copy();y=y.copy()
    excluded=ignored_region(y.shape,box);y[excluded]=IGNORE
    if excluded.any():rgb[excluded]=np.median(rgb[~excluded],axis=0).astype(np.uint8)
    h,w=y.shape;scale=min(size/w,size/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    l=(size-nw)//2;t=(size-nh)//2
    x=np.zeros((size,size,3),np.uint8);target=np.full((size,size),IGNORE,np.uint8)
    resample=Image.Resampling.BOX if scale<1 else Image.Resampling.BILINEAR
    x[t:t+nh,l:l+nw]=np.array(Image.fromarray(rgb).resize((nw,nh),resample))
    target[t:t+nh,l:l+nw]=np.array(Image.fromarray(y).resize((nw,nh),Image.Resampling.NEAREST))
    return x,target,{'top':t,'left':l,'height':nh,'width':nw,'original_height':h,'original_width':w}


def make_model(base=16):
    import torch
    from torch import nn
    class Double(nn.Module):
        def __init__(self,a,b):
            super().__init__();self.layers=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        def forward(self,x):return self.layers(x)
    class UNet(nn.Module):
        def __init__(self):
            super().__init__();b=base
            self.e1=Double(3,b);self.e2=Double(b,2*b);self.e3=Double(2*b,4*b);self.mid=Double(4*b,8*b)
            self.pool=nn.MaxPool2d(2)
            self.u3=nn.ConvTranspose2d(8*b,4*b,2,2);self.d3=Double(8*b,4*b)
            self.u2=nn.ConvTranspose2d(4*b,2*b,2,2);self.d2=Double(4*b,2*b)
            self.u1=nn.ConvTranspose2d(2*b,b,2,2);self.d1=Double(2*b,b)
            self.out=nn.Conv2d(b,4,1)
        def forward(self,x):
            a=self.e1(x);b=self.e2(self.pool(a));c=self.e3(self.pool(b));z=self.mid(self.pool(c))
            z=self.d3(torch.cat((self.u3(z),c),1));z=self.d2(torch.cat((self.u2(z),b),1));z=self.d1(torch.cat((self.u1(z),a),1))
            return self.out(z)
    return UNet()


def loss_fn(logits,target,weights):
    import torch
    valid=target!=IGNORE
    if not valid.any(): return logits.sum()*0
    ce=torch.nn.functional.cross_entropy(logits,target,weight=weights,ignore_index=IGNORE)
    safe=target.masked_fill(~valid,0)
    onehot=torch.nn.functional.one_hot(safe,4).permute(0,3,1,2).float()
    v=valid[:,None];p=logits.softmax(1)
    intersection=(p*onehot*v).sum((0,2,3))
    denominator=((p+onehot)*v).sum((0,2,3))
    # Include all foreground classes; absent classes are still penalized by CE.
    dice=(2*intersection+1)/(denominator+1)
    return ce+(1-dice[1:]).mean()


def confusion(pred,y):
    valid=y!=IGNORE
    return np.bincount((y[valid]*4+pred[valid]).ravel(),minlength=16).reshape(4,4)


def scores(cm):
    rows=[]
    for k,name in enumerate(NAMES):
        tp=int(cm[k,k]);truth=int(cm[k].sum());pred=int(cm[:,k].sum());total=truth+pred;union=total-tp
        rows.append({'class':name,'dice':2*tp/total if total else None,'iou':tp/union if union else None,
                     'target_pixels':truth,'predicted_pixels':pred})
    defined=[r['dice'] for r in rows[1:] if r['dice'] is not None]
    return rows,float(np.mean(defined)) if defined else 0.


def restore(pred,meta):
    t,l,h,w=[meta[k] for k in ('top','left','height','width')]
    return np.array(Image.fromarray(pred[t:t+h,l:l+w]).resize((meta['original_width'],meta['original_height']),Image.Resampling.NEAREST))


def train(args):
    if not args.labels_reviewed:
        raise ValueError('Review/correct prepared labels first, then add --labels-reviewed. Candidate conversion is not verified ground truth.')
    import torch
    from torch.utils.data import Dataset,DataLoader
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=Path(args.data);pairs=json.loads((folder/'pairs.json').read_text());prep=json.loads((folder/'preparation.json').read_text())
    n=len(pairs)
    if args.val_count+args.test_count>n-2:raise ValueError('Need at least two training photographs')
    if args.split:
        split=json.loads(Path(args.split).read_text());ids={p['id'] for p in pairs}
        flat=[key for role in ['train','validation','test'] for key in split[role]]
        if len(flat)!=len(set(flat)) or not ids.issubset(set(flat)) or any(not split[r] for r in ['train','validation','test']):
            raise ValueError('Supplied split must contain every ID once with three nonempty sets')
        index={p['id']:i for i,p in enumerate(pairs)}
        split_idx={r:[index[k] for k in split[r] if k in index] for r in ['train','validation','test']}
    else:
        order=np.random.default_rng(args.seed).permutation(n).tolist()
        split_idx={'test':order[:args.test_count],'validation':order[args.test_count:args.test_count+args.val_count],
                   'train':order[args.test_count+args.val_count:]}
        split={r:[pairs[i]['id'] for i in ix] for r,ix in split_idx.items()}
    if any(not split_idx[r] for r in ['train','validation','test']):
        raise ValueError('Every split needs at least one masked image; adjust the plan before training.')
    out=fresh(args.out);dump(out/'splits.json',split)
    prepared=[];manifest=[]
    (out/'input_previews').mkdir()
    for p in pairs:
        ip=folder/p['image'];lp=folder/p['label'];rgb=load_image(ip);y=decode(load_image(lp,True))
        if rgb.shape[:2]!=y.shape:raise ValueError(f'Mismatched dimensions: {p["id"]}')
        x,target,meta=model_input(rgb,y,args.size,prep['ignore_box']);prepared.append((x,target,meta))
        preview(out/'input_previews'/f'{p["id"]}.jpg',x,target,'Actual resized training/evaluation input')
        manifest.append({'id':p['id'],'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),
                         'label_sha256':hashlib.sha256(lp.read_bytes()).hexdigest()})
    counts=sum((np.bincount(prepared[i][1][prepared[i][1]!=IGNORE],minlength=4) for i in split_idx['train']),np.zeros(4,dtype=np.int64))
    if (counts==0).any():raise ValueError(f'Training split lacks classes: {[NAMES[k] for k in np.where(counts==0)[0]]}. Review labels/split before training.')
    # Training-only frequencies; capped to avoid extreme rare-class weights.
    weights=np.sqrt(counts.sum()/(4*counts));weights=np.clip(weights,.25,4).astype(np.float32)
    weight_tensor=torch.tensor(weights,device=device)
    config={**vars(args),'classes':NAMES,'ignore_box':prep['ignore_box'],'class_weights':weights.tolist(),
            'training_originals':len(split_idx['train']),'training_views_per_epoch':len(split_idx['train'])*args.rotations,
            'device':device,'torch':str(torch.__version__),'metric_grid':'resized real pixels, excluding padding/ignore regions'}
    dump(out/'config.json',config);csvwrite(out/'data_manifest.csv',manifest)
    shutil.copy2(__file__,out/'segmentation_source.py')
    class Cells(Dataset):
        def __init__(self,indices,rotations=1):self.indices=indices;self.rotations=rotations
        def __len__(self):return len(self.indices)*self.rotations
        def __getitem__(self,j):
            idx=self.indices[j//self.rotations];k=j%self.rotations
            x,y,_=prepared[idx]
            x=np.rot90(x,k);y=np.rot90(y,k)
            return (torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()/255,
                    torch.from_numpy(np.ascontiguousarray(y)).long(),idx)
    def loader(indices,training=False):
        return DataLoader(Cells(indices,args.rotations if training else 1),batch_size=args.batch_size,shuffle=training,num_workers=0)
    @torch.no_grad()
    def evaluate(model,indices,save=False):
        model.eval();cm=np.zeros((4,4),np.int64);rows=[]
        if save:(out/'test_predictions').mkdir()
        for x,y,idxs in loader(indices):
            predictions=model(x.to(device)).argmax(1).cpu().numpy()
            for j,idx in enumerate(idxs.tolist()):
                truth=y[j].numpy();pred=predictions[j].astype(np.uint8);local=confusion(pred,truth);cm+=local
                class_rows,_=scores(local)
                for row in class_rows:rows.append({'image_id':pairs[idx]['id'],**row})
                if save:
                    pred[truth==IGNORE]=IGNORE
                    key=pairs[idx]['id'];meta=prepared[idx][2]
                    Image.fromarray(encode(pred)).save(out/'test_predictions'/f'{key}_labels_model_grid.png')
                    Image.fromarray(encode(restore(pred,meta))).save(out/'test_predictions'/f'{key}_labels_original_size.png')
                    preview(out/'test_predictions'/f'{key}_prediction.jpg',prepared[idx][0],pred,f'Foto {key}: PREDICTION')
        return cm,rows
    model=make_model(args.base).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=-1.;best_epoch=0;stale=0;history=[]
    train_loader=loader(split_idx['train'],True)
    print(f'{n} originals: {len(split_idx["train"])} train / {len(split_idx["validation"])} validation / {len(split_idx["test"])} test')
    print(f'{len(train_loader.dataset)} training views per epoch; device={device}')
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;seen=0
        for x,y,_ in train_loader:
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y,weight_tensor)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(x);seen+=len(x)
        cm,_=evaluate(model,split_idx['validation']);_,score=scores(cm)
        history.append({'epoch':epoch,'training_loss':total/seen,'validation_macro_foreground_dice':score})
        csvwrite(out/'history.csv',history)
        if score>best:
            best=score;best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'base':args.base,'size':args.size,'classes':NAMES,
                        'ignore_box':prep['ignore_box'],'epoch':epoch,'experiment':'C',
                        'splits':split,'classification_labels':{'0':'normal','1':'anomaly'}},out/'best_model.pt')
        else:stale+=1
        print(f'epoch={epoch} loss={total/seen:.4f} validation_Dice={score:.4f}',flush=True)
        if stale>=args.patience:break
    state=torch.load(out/'best_model.pt',map_location=device,weights_only=True);model.load_state_dict(state['model'])
    cm,rows=evaluate(model,split_idx['test'],True);class_rows,macro=scores(cm)
    csvwrite(out/'test_per_image.csv',rows);csvwrite(out/'test_per_class.csv',class_rows);dump(out/'test_confusion_matrix.json',cm.tolist())
    # Common whole-cell union = cytoplasm + nucleus, excluding pink regions.
    cell=[1,2];tp=int(cm[np.ix_(cell,cell)].sum());truth=int(cm[cell,:].sum());pred=int(cm[:,cell].sum())
    summary={'best_epoch':best_epoch,'best_validation_macro_foreground_dice':best,
             'test_macro_foreground_dice':macro,'test_images':len(split_idx['test']),
             'whole_cell_dice':2*tp/(truth+pred) if truth+pred else None,
             'whole_cell_iou':tp/(truth+pred-tp) if truth+pred-tp else None,
             'note':'Pooled pixels on resized images. These are not Experiment 2 per-image means.'}
    dump(out/'summary.json',summary);dump(out/'status.json',{'status':'completed'})
    print(json.dumps(summary,indent=2))


def predict(args):
    import torch
    torch.set_num_threads(args.threads);device='cuda' if torch.cuda.is_available() else 'cpu'
    state=torch.load(args.model,map_location=device,weights_only=True)
    if state.get('experiment')!='C':raise ValueError('Use an Experiment C checkpoint')
    model=make_model(state['base']).to(device);model.load_state_dict(state['model']);model.eval()
    files=sorted(p for p in Path(args.images).rglob('*') if p.suffix.lower() in EXTS and 'mask' not in p.stem.lower())
    if not files:raise ValueError('No photographs found')
    out=fresh(args.out);ids=set();rows=[]
    with torch.no_grad():
        for file in files:
            key=photo_id(file)
            if key in ids:raise ValueError(f'Duplicate ID {key}')
            ids.add(key);rgb=load_image(file);dummy=np.zeros(rgb.shape[:2],np.uint8)
            x,y,meta=model_input(rgb,dummy,state['size'],state['ignore_box'])
            tensor=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()[None]/255
            pred=model(tensor.to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8);pred[y==IGNORE]=IGNORE
            original=restore(pred,meta);original[ignored_region(original.shape,state['ignore_box'])]=IGNORE
            Image.fromarray(encode(original)).save(out/f'{key}_labels.png')
            Image.fromarray(colorize(original)).save(out/f'{key}_colors.png')
            preview(out/f'{key}_prediction.jpg',rgb,original,f'Foto {key}: PREDICTION')
            for k,name in enumerate(NAMES[1:],1):Image.fromarray((original==k).astype(np.uint8)*255).save(out/f'{key}_{name}.png')
            Image.fromarray(np.isin(original,[1,2]).astype(np.uint8)*255).save(out/f'{key}_whole_cell.png')
            Image.fromarray((original!=IGNORE).astype(np.uint8)*255).save(out/f'{key}_valid_region.png')
            rows.append({'image_id':key,'original':str(file.resolve()),'model':str(Path(args.model).resolve())})
    csvwrite(out/'prediction_manifest.csv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    a=sp.add_parser('prepare',help='Convert supplied ZIP to editable candidate labels')
    a.add_argument('--zip',required=True);a.add_argument('--out',default='C_data')
    a.add_argument('--ignore-box',type=float,nargs=4,default=[.76,.86,1.,1.],metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    a.add_argument('--keep-timestamp',action='store_true',help='Disable default ignored timestamp rectangle')
    a=sp.add_parser('train',help='Train on reviewed four-class labels')
    a.add_argument('--data',default='C_data');a.add_argument('--out',default='C_results');a.add_argument('--labels-reviewed',action='store_true')
    a.add_argument('--size',type=int,default=768);a.add_argument('--epochs',type=int,default=50);a.add_argument('--patience',type=int,default=12)
    a.add_argument('--batch-size',type=int,default=2);a.add_argument('--base',type=int,default=16);a.add_argument('--lr',type=float,default=.001)
    a.add_argument('--seed',type=int,default=42);a.add_argument('--val-count',type=int,default=4);a.add_argument('--test-count',type=int,default=5)
    a.add_argument('--rotations',type=int,choices=[1,4],default=4);a.add_argument('--split',help='Reuse a saved splits.json on the same IDs')
    a.add_argument('--threads',type=int,default=4)
    a=sp.add_parser('predict',help='Predict new photographs')
    a.add_argument('--model',default='C_results/best_model.pt');a.add_argument('--images',required=True);a.add_argument('--out',default='C_predictions');a.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if args.command=='prepare':
        if args.keep_timestamp:args.ignore_box=None
        if args.ignore_box is not None:
            l,t,r,b=args.ignore_box
            if not (0<=l<r<=1 and 0<=t<b<=1):p.error('Ignore box needs 0 <= left < right <= 1 and 0 <= top < bottom <= 1')
    if args.command=='train':
        if args.size<32 or args.size%8:p.error('size must be >=32 and divisible by 8')
        if min(args.epochs,args.patience,args.batch_size,args.base,args.val_count,args.test_count,args.threads)<=0 or args.lr<=0:p.error('Training counts and learning rate must be positive')
    if args.command=='predict' and args.threads<1:p.error('threads must be positive')
    {'prepare':prepare,'train':train,'predict':predict}[args.command](args)


if __name__=='__main__':main()
