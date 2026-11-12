"""Apply a trained MIL classifier and its frozen threshold to extracted bags."""
import argparse
import json
from pathlib import Path
import torch
from train_mil import AttentionMIL, evaluate
from segmentation import fresh, dump, csvwrite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bags',required=True)
    p.add_argument('--model',required=True,help='best_mil.pt')
    p.add_argument('--out',default='C_new_predictions')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.threads<1:p.error('threads must be positive')
    torch.set_num_threads(a.threads)
    root=Path(a.bags);bags=json.loads((root/'bags.json').read_text())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    state=torch.load(a.model,map_location=device,weights_only=True)
    if state['classification_labels']!={'0':'normal','1':'anomaly'}:raise ValueError('Unexpected label mapping')
    if state['segmentation_model_sha256']!=bags['segmentation_model_sha256']:
        raise ValueError('Use the same segmentation checkpoint used to create classifier training bags')
    model=AttentionMIL().to(device);model.load_state_dict(state['model'])
    rows,attention=evaluate(model,bags['records'],root,device)
    for r in rows:
        prob=r['probability_anomaly']
        label=int(prob>=state['threshold']) if prob is not None else None
        r['predicted_label']=label
        r['predicted_class']='abstain' if label is None else ('anomaly' if label else 'normal')
    out=fresh(a.out);csvwrite(out/'predictions.csv',rows);csvwrite(out/'attention.csv',attention)
    dump(out/'settings.json',{'threshold':state['threshold'],'classification_labels':state['classification_labels']})
    print('Saved',out/'predictions.csv')

if __name__=='__main__':main()
