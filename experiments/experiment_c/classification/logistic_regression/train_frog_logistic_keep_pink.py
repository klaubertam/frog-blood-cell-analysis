"""Run from Experiment_C_Normal_Anomaly_Improved after the all-photo evaluation."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Fixed before running. Summarize typical cells and unusually shaped cells.
FEATURES = {
    "cell_axis_ratio": (.5, .9),
    "cell_solidity": (.5, .1),
    "nuclear_to_cell_area": (.5, .9),
}
NAMES = [f"{key}_q{int(q*100)}" for key, qs in FEATURES.items() for q in qs]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def indexed(rows, key):
    result = {}
    for row in rows:
        foto = int(row[key])
        if foto in result:
            raise ValueError(f"Duplicate Foto {foto}")
        result[foto] = row
    return result


def measurements(cells):
    values = []
    for key, quantiles in FEATURES.items():
        numbers = [float(row[key]) for row in cells if row[key].strip()]
        finite = np.asarray([x for x in numbers if np.isfinite(x)])
        values.extend(np.quantile(finite, quantiles) if finite.size else [np.nan]*len(quantiles))
    return values


def estimator():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1, class_weight="balanced",
                         solver="liblinear", max_iter=2000, random_state=42))


def metrics(y, pred, scores=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return dict(n=len(y), correct=tp+tn, accuracy=(tp+tn)/len(y),
                balanced_accuracy=(sensitivity+specificity)/2, sensitivity=sensitivity,
                specificity=specificity, tn=tn, fp=fp, fn=fn, tp=tp,
                auc=float(roc_auc_score(y, scores)) if scores is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("C_logistic"))
    args = parser.parse_args()
    bags = Path("C_bags_all_keep_pink")
    manifest = json.loads((bags/"bags.json").read_text())
    photos = indexed(manifest["records"], "id")
    labels = indexed(read_rows(Path("C_predictions_all_photos/accuracy_by_photo.csv")), "foto")
    if photos.keys() != labels.keys():
        raise ValueError("Foto IDs differ between bags and the Excel comparison CSV.")
    split = json.loads(Path("C_plan/splits.json").read_text())
    train = set(map(int, split["train"]))
    pilot = {int(i) for role in ("train","validation","test") for i in split[role]}
    if not pilot <= photos.keys():
        raise ValueError("Pilot photos missing from all-photo bags.")
    cells = {i: [] for i in photos}
    for row in read_rows(bags/"cell_features.csv"):
        cells[int(row["id"])].append(row)
    ids = sorted(photos)
    counts = np.array([len(cells[i]) for i in ids])
    if any(int(photos[i]["instances"]) != len(cells[i]) for i in ids):
        raise ValueError("Cell feature counts disagree with bags.json.")
    X = np.array([measurements(cells[i]) for i in ids])
    y = np.array([int(labels[i]["excel_label"]) for i in ids])
    if not set(y) <= {0,1}:
        raise ValueError("Expected Excel labels 0=normal and 1=anomaly.")
    outside = np.array([i not in pilot for i in ids])
    usable = (counts > 0) & np.isfinite(X).any(axis=1)
    base = np.flatnonzero(usable & np.isin(ids, list(train)))
    test = np.flatnonzero(usable & outside)
    if len(set(y[base])) != 2 or np.bincount(y[test], minlength=2).min() < 5:
        raise ValueError("Too few usable labeled photographs per class.")
    fixed = estimator().fit(X[base], y[base]).predict_proba(X[test])[:,1]
    scores = np.empty(len(test))
    folds = np.zeros(len(test), dtype=int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(cv.split(X[test], y[test]), 1):
        fit = np.concatenate([base, test[tr]])
        model = estimator().fit(X[fit], y[fit])
        scores[va] = model.predict_proba(X[test[va]])[:,1]
        folds[va] = fold
    majority = int(np.bincount(y[base], minlength=2).argmax())
    summary = dict(outside_original_30=int(outside.sum()), scored=len(test),
                   abstentions=int(outside.sum()-len(test)), coverage=len(test)/int(outside.sum()),
                   pilot_training_photos=len(base), threshold=.5,
                   majority=metrics(y[test], np.full(len(test),majority)),
                   logistic_same_training_photos=metrics(y[test],fixed>=.5,fixed),
                   logistic_more_training_photos_5fold=metrics(y[test],scores>=.5,scores))
    # This last fit is for future photos. Its training scores are never reported.
    final = estimator().fit(X[usable], y[usable])
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    joblib.dump(dict(pipeline=final, features=NAMES, quantiles=FEATURES, threshold=.5,
                     fitted_photo_ids=[i for i,u in zip(ids,usable) if u]), args.out/"logistic_model.joblib")
    position = {j:k for k,j in enumerate(test)}
    with (args.out/"outside_predictions.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.writer(handle)
        writer.writerow(["foto","excel_label","cells","status","pilot_lr_score","pilot_lr_label",
                         "cv_fold","cv_lr_score","cv_lr_label"])
        for j in np.flatnonzero(outside):
            k=position.get(j)
            tail=["abstain","","","","",""] if k is None else [
                "classified",fixed[k],int(fixed[k]>=.5),folds[k],scores[k],int(scores[k]>=.5)]
            writer.writerow([ids[j],y[j],counts[j],*tail])
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
