"""Train an attention MIL classifier. One photograph = one bag = one target."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from segmentation import fresh, dump, csvwrite


class AttentionMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.attention = nn.Sequential(nn.Linear(64,32), nn.Tanh(), nn.Linear(32,1))
        self.head = nn.Sequential(nn.Dropout(.25), nn.Linear(64,1))

    def forward(self, cells):
        if len(cells) == 0:
            raise ValueError('Empty bags require abstention, not a fabricated prediction')
        features = self.encoder(cells)
        weights = self.attention(features).softmax(dim=0)
        pooled = (weights*features).sum(dim=0)
        return self.head(pooled).squeeze(-1), weights[:,0]


def select_threshold(y, probability):
    if set(y) != {0,1}:
        raise ValueError('Threshold selection needs both classes in validation')
    fpr,tpr,thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    thresholds, j = thresholds[valid], (tpr-fpr)[valid]
    best = np.flatnonzero(np.isclose(j, j.max()))
    idx = best[np.argmin(abs(thresholds[best]-.5))]
    return float(thresholds[idx])


def report(rows, threshold):
    usable = [r for r in rows if r['probability_anomaly'] is not None]
    result = {'total_images':len(rows), 'classified_images':len(usable),
              'coverage':len(usable)/len(rows) if rows else 0, 'threshold':threshold}
    if not usable:
        return result
    y = np.array([r['target'] for r in usable])
    prob = np.array([r['probability_anomaly'] for r in usable])
    pred = prob >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    sensitivity = float(tp/(tp+fn)) if tp+fn else None
    specificity = float(tn/(tn+fp)) if tn+fp else None
    result.update({'confusion_matrix_rows_true_0_1':[[int(tn),int(fp)],[int(fn),int(tp)]],
        'accuracy_classified':float((pred==y).mean()),
        'correct_predictions_per_all_images':float((pred==y).sum()/len(rows)),
        'sensitivity_anomaly':sensitivity,'specificity_normal':specificity,
        'balanced_accuracy':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
        'precision_anomaly':float(tp/(tp+fp)) if tp+fp else 0.,
        'f1_anomaly':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
        'roc_auc':float(roc_auc_score(y,prob)) if len(set(y))==2 else None})
    return result


def get_bag(root, record, training=False, max_cells=128):
    with np.load(root/record['file']) as z:
        crops = z['crops'].copy()
        ids = z['instance_ids'].copy()
    if training and len(crops)>max_cells:
        selection = np.random.choice(len(crops),max_cells,replace=False)
        crops,ids = crops[selection],ids[selection]
    if training:
        crops = np.rot90(crops,random.randrange(4),axes=(1,2))
    tensor = torch.from_numpy(crops.transpose(0,3,1,2).copy()).float()/255
    return tensor,ids


@torch.no_grad()
def evaluate(model, records, root, device):
    model.eval(); rows=[]; attention=[]
    for r in records:
        x,ids = get_bag(root,r)
        if len(x):
            logit, weights = model(x.to(device))
            probability = float(logit.sigmoid().cpu())
            attention.extend({'id':r['id'],'instance_id':int(i),'attention_weight':float(w)}
                             for i,w in zip(ids,weights.cpu().numpy()))
        else:
            probability = None
        rows.append({'id':r['id'],'target':r['target'],'instances':len(x),
                     'probability_anomaly':probability,
                     'status':'classified' if len(x) else 'ABSTAIN: no retained cells'})
    return rows,attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bags',required=True,help='Folder created by extract_bags.py')
    p.add_argument('--out',default='C_classifier')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--max-training-cells',type=int,default=128)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.epochs,a.patience,a.max_training_cells,a.threads)<1 or a.lr<=0:
        p.error('Counts and learning rate must be positive')
    torch.set_num_threads(a.threads)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(a.bags);data=json.loads((root/'bags.json').read_text())
    if data['classification_labels']!={'0':'normal','1':'anomaly'}:
        raise ValueError('Classification label mapping is not normal=0, anomaly=1')
    records=data['records'];all_ids=[r['id'] for r in records]
    if len(set(all_ids))!=len(all_ids):raise ValueError('Duplicate bag IDs')
    parts={role:[r for r in records if r['split']==role] for role in ['train','validation','test']}
    for role,rs in parts.items():
        if {r['id'] for r in rs}!=set(data['splits'][role]):raise ValueError('Bag split disagrees with saved plan')
    flat=[i for keys in data['splits'].values() for i in keys]
    if len(set(flat))!=len(flat):raise ValueError('Image leakage across splits')
    train=[r for r in parts['train'] if r['instances']>0]
    val=[r for r in parts['validation'] if r['instances']>0]
    for name,rs in [('training',train),('validation',val)]:
        if {r['target'] for r in rs}!={0,1}:
            raise ValueError(f'{name} needs usable normal AND anomaly bags; inspect segmentation/ROI quality')
    out=fresh(a.out);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AttentionMIL().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    counts=np.bincount([r['target'] for r in train],minlength=2)
    class_weights=len(train)/(2*counts)
    dump(out/'config.json',{**vars(a),'class_weights':class_weights.tolist(),
        'label_mapping':{'0':'normal','1':'anomaly'},'splits':data['splits'],
        'segmentation_model_sha256':data['segmentation_model_sha256'],
        'torch_version':str(torch.__version__),'numpy_version':np.__version__})
    history=[];best=float('inf');stale=0
    for epoch in range(1,a.epochs+1):
        model.train();random.shuffle(train);losses=[]
        for r in train:
            x,_=get_bag(root,r,True,a.max_training_cells)
            target=torch.tensor(float(r['target']),device=device)
            optimizer.zero_grad(set_to_none=True)
            logit,_=model(x.to(device))
            loss=nn.functional.binary_cross_entropy_with_logits(logit,target)*float(class_weights[r['target']])
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val_rows,_=evaluate(model,val,root,device)
        yy=np.array([r['target'] for r in val_rows])
        pp=np.clip([r['probability_anomaly'] for r in val_rows],1e-7,1-1e-7)
        val_loss=float(np.mean((-yy*np.log(pp)-(1-yy)*np.log1p(-pp))*class_weights[yy]))
        history.append({'epoch':epoch,'train_weighted_bce':float(np.mean(losses)),
                        'validation_weighted_bce':val_loss})
        csvwrite(out/'history.csv',history)
        if val_loss<best:
            best=val_loss;stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'classification_labels':{'0':'normal','1':'anomaly'},
                         'splits':data['splits'],'segmentation_model_sha256':data['segmentation_model_sha256']},out/'best_mil.pt')
        else:stale+=1
        print(f'epoch={epoch} train_BCE={np.mean(losses):.4f} validation_BCE={val_loss:.4f}',flush=True)
        if stale>=a.patience:break
    state=torch.load(out/'best_mil.pt',map_location=device,weights_only=True)
    model.load_state_dict(state['model'])
    val_rows,_=evaluate(model,val,root,device)
    threshold=select_threshold([r['target'] for r in val_rows],[r['probability_anomaly'] for r in val_rows])
    state['threshold']=threshold;torch.save(state,out/'best_mil.pt')
    test_rows,attention=evaluate(model,parts['test'],root,device)
    for r in test_rows:
        r['predicted_label']=int(r['probability_anomaly']>=threshold) if r['probability_anomaly'] is not None else None
    csvwrite(out/'validation_predictions.csv',val_rows)
    csvwrite(out/'test_predictions.csv',test_rows);csvwrite(out/'test_attention.csv',attention)
    result=report(test_rows,threshold)
    result['metrics_at_0_5']=report(test_rows,.5)
    majority=int(counts.argmax())
    result['training_majority_class']=majority
    result['majority_baseline_test_accuracy']=sum(r['target']==majority for r in parts['test'])/len(parts['test'])
    result['note']='Held-out image-level pilot. Attention is influence, not proof that a cell is abnormal. Scores are not calibrated exposure probabilities.'
    dump(out/'test_metrics.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
