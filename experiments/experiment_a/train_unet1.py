"""Train a binary U-Net on paired frog erythrocyte images and whole-cell masks.
Usage: python train_unet.py --images images --masks masks --out results --epochs 40 --folds 5
Dependencies: pip install torch numpy pillow opencv-python scikit-learn
"""
import argparse, csv, random, re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold

def identifier(path):
    m = re.search(r'(?:foto)[_\s-]*(\d+)', path.stem.lower())
    return str(int(m.group(1))) if m else path.stem.lower().replace('_cell_mask','').replace('_mask','')

def find_pairs(images, masks):
    extensions={'.png','.jpg','.jpeg','.tif','.tiff'}
    ims={identifier(p):p for p in Path(images).rglob('*') if p.suffix.lower() in extensions}
    ms={identifier(p):p for p in Path(masks).rglob('*') if p.suffix.lower() in extensions and 'overlay' not in p.stem.lower()}
    pairs=[(ims[k],ms[k],k) for k in sorted(ims.keys() & ms.keys(),key=lambda x:int(x) if x.isdigit() else x)]
    if not pairs: raise ValueError('No matching Foto numbers found between images and masks')
    print(f'Matched {len(pairs)} pairs; unmatched images={len(ims)-len(pairs)}, unmatched masks={len(ms)-len(pairs)}')
    return pairs

class BloodDataset(Dataset):
    def __init__(self,pairs,augment=False,clahe=False,size=256):
        self.pairs,self.augment,self.clahe,self.size=pairs,augment,clahe,size
    def __len__(self):return len(self.pairs)
    def __getitem__(self,i):
        ip,mp,_=self.pairs[i]
        rgb=np.asarray(Image.open(ip).convert('RGB'))
        mask=np.asarray(Image.open(mp).convert('L'))
        if rgb.shape[:2]!=mask.shape[:2]:
            raise ValueError(f'Geometry mismatch: {ip} vs {mp}; align masks BEFORE training')
        # Exclude fixed bottom-right timestamp by cropping SAME region from image and mask.
        # Assumes timestamp is in bottom-right; verify on all images.
        h,w=mask.shape
        rgb=rgb[:int(h*.86),:int(w*.76)]
        mask=mask[:int(h*.86),:int(w*.76)]
        rgb=cv2.resize(rgb,(self.size,self.size),interpolation=cv2.INTER_AREA)
        mask=cv2.resize(mask,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        # Masks must be binary whole-cell targets, not colored class masks.
        mask=(mask>127).astype(np.uint8)
        if self.clahe:
            lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
            lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
            rgb=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
        if self.augment:
            angle=random.uniform(-20,20); scale=random.uniform(.9,1.1)
            mat=cv2.getRotationMatrix2D((self.size/2,self.size/2),angle,scale)
            mat[0,2]+=random.uniform(-.05,.05)*self.size
            mat[1,2]+=random.uniform(-.05,.05)*self.size
            rgb=cv2.warpAffine(rgb,mat,(self.size,self.size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
            mask=cv2.warpAffine(mask,mat,(self.size,self.size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            if random.random()<.5:rgb=np.fliplr(rgb);mask=np.fliplr(mask)
            if random.random()<.5:rgb=np.flipud(rgb);mask=np.flipud(mask)
        x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255.
        y=torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return x,y

class DoubleConv(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x):return self.net(x)
class UNet(nn.Module):
    def __init__(self,base=32):
        super().__init__();b=base
        self.e1=DoubleConv(3,b);self.e2=DoubleConv(b,2*b);self.e3=DoubleConv(2*b,4*b);self.e4=DoubleConv(4*b,8*b)
        self.pool=nn.MaxPool2d(2);self.mid=DoubleConv(8*b,16*b)
        self.u4=nn.ConvTranspose2d(16*b,8*b,2,stride=2);self.d4=DoubleConv(16*b,8*b)
        self.u3=nn.ConvTranspose2d(8*b,4*b,2,stride=2);self.d3=DoubleConv(8*b,4*b)
        self.u2=nn.ConvTranspose2d(4*b,2*b,2,stride=2);self.d2=DoubleConv(4*b,2*b)
        self.u1=nn.ConvTranspose2d(2*b,b,2,stride=2);self.d1=DoubleConv(2*b,b)
        self.out=nn.Conv2d(b,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(self.pool(a));c=self.e3(self.pool(b));d=self.e4(self.pool(c));z=self.mid(self.pool(d))
        z=self.d4(torch.cat([self.u4(z),d],1));z=self.d3(torch.cat([self.u3(z),c],1))
        z=self.d2(torch.cat([self.u2(z),b],1));z=self.d1(torch.cat([self.u1(z),a],1))
        return self.out(z)

def loss_fn(logits,y):
    bce=nn.functional.binary_cross_entropy_with_logits(logits,y)
    p=torch.sigmoid(logits);dims=(1,2,3)
    dice=1-(2*(p*y).sum(dims)+1)/((p+y).sum(dims)+1)
    return bce+dice.mean()

def metrics(model,loader,device,save_dir=None):
    model.eval();scores=[]
    if save_dir:Path(save_dir).mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            p=(torch.sigmoid(model(x.to(device))).cpu()>0.5).float()
            intersection=(p*y).sum((1,2,3)); union=(p+y-p*y).sum((1,2,3))
            dice=((2*intersection+1e-7)/(p.sum((1,2,3))+y.sum((1,2,3))+1e-7)).numpy()
            iou=((intersection+1e-7)/(union+1e-7)).numpy()
            scores.extend(zip(dice,iou))
            if save_dir:
                for j,mask in enumerate(p):Image.fromarray((mask[0].numpy()*255).astype(np.uint8)).save(Path(save_dir)/f'prediction_{i*loader.batch_size+j:04d}.png')
    return np.mean(scores,axis=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--images',required=True);ap.add_argument('--masks',required=True);ap.add_argument('--out',default='results')
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--folds',type=int,default=5);ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--clahe',action='store_true');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--groups-csv',help='Optional CSV with columns foto_number,animal_id for animal-level grouped folds')
    args=ap.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    pairs=find_pairs(args.images,args.masks)
    if len(pairs)<args.folds:raise ValueError('Need at least as many pairs as folds')
    groups=None
    if args.groups_csv:
        with open(args.groups_csv,newline='') as f:groups_map={str(int(r['foto_number'])):r['animal_id'] for r in csv.DictReader(f)}
        groups=[groups_map[p[2]] for p in pairs]
        if len(set(groups))<args.folds:raise ValueError('Not enough distinct animals for grouped folds')
        splits=GroupKFold(n_splits=args.folds).split(pairs,groups=groups)
    else:
        print('WARNING: image-level folds only. If images share animals/slides, provide --groups-csv to prevent leakage.')
        splits=KFold(n_splits=args.folds,shuffle=True,random_state=args.seed).split(pairs)
    device='cuda' if torch.cuda.is_available() else 'cpu';print('Device:',device)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for fold,(tr,va) in enumerate(splits,1):
        train=DataLoader(BloodDataset([pairs[i] for i in tr],True,args.clahe),batch_size=args.batch_size,shuffle=True,num_workers=0)
        valid=DataLoader(BloodDataset([pairs[i] for i in va],False,args.clahe),batch_size=args.batch_size,shuffle=False,num_workers=0)
        model=UNet().to(device);opt=torch.optim.Adam(model.parameters(),lr=args.lr);best=-1
        for epoch in range(1,args.epochs+1):
            model.train();losses=[]
            for x,y in train:
                x,y=x.to(device),y.to(device);opt.zero_grad();loss=loss_fn(model(x),y);loss.backward();opt.step();losses.append(loss.item())
            dice,iou=metrics(model,valid,device)
            if dice>best:
                best=dice;torch.save({'model':model.state_dict(),'clahe':args.clahe,'size':256},out/f'unet_fold_{fold}.pt')
            print(f'fold={fold} epoch={epoch} train_loss={np.mean(losses):.4f} val_dice={dice:.4f} val_iou={iou:.4f}')
        state=torch.load(out/f'unet_fold_{fold}.pt',map_location=device,weights_only=True);model.load_state_dict(state['model'])
        dice,iou=metrics(model,valid,device,out/f'fold_{fold}_predictions');rows.append({'fold':fold,'dice':float(dice),'iou':float(iou)})
    with open(out/'segmentation_cv.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['fold','dice','iou']);w.writeheader();w.writerows(rows)
    print('Saved:',out/'segmentation_cv.csv','mean Dice',np.mean([r['dice'] for r in rows]))
if __name__=='__main__':main()
