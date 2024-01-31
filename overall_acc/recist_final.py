import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import numpy as np

status = "Ts"

gt = pd.read_csv(f"gt_tla_ntla_{status.lower()}.csv", index_col="file")
pred = pd.read_csv(f"pred_tla_ntla_{status.lower()}.csv", index_col="file")

img_path = "images{status}/"
files = os.listdir(img_path)
files = [x.replace("_0000", "") for x in files]

if status == "Val":
    patients = list(set([x[:9] for x in files]))
else:
    patients = list(set([x[:8] for x in files]))


def recist_outcome(tla_baseline, tla_followup, ntla_baseline, ntla_followup):
    if max(tla_baseline, tla_followup) < 10:
        return 1
    else:
        tla_ratio = tla_followup / (tla_baseline + 0.001)
        ntla_ratio = ntla_followup / (ntla_baseline + 0.001)
        if (tla_ratio < 0.7) and (ntla_ratio < 1.2):
            return 0
        elif (tla_ratio > 0.7) and (tla_ratio < 1.2) and (ntla_ratio < 1.2):
            return 1
        else:
            return 2


results = []

for patient in tqdm(patients):
    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    if status == "Ts":
        baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[9:17], "%Y%m%d"))
    else:
        baseline = min(patient_visits, key=lambda visit: int(visit.split(".")[0].split("-")[1]))
    patient_visits.remove(baseline)
    tla_gt = gt.loc[baseline.replace(".nii.gz", ""), "tla"]
    ntla_gt = gt.loc[baseline.replace(".nii.gz", ""), "ntla"]

    tla_pred = pred.loc[baseline.replace(".nii.gz", ""), "tla"]
    ntla_pred = pred.loc[baseline.replace(".nii.gz", ""), "ntla"]
    for followup_visit in patient_visits:
        tla_f_gt = gt.loc[followup_visit.replace(".nii.gz", ""), "tla"]
        ntla_f_gt = gt.loc[followup_visit.replace(".nii.gz", ""), "ntla"]

        tla_f_pred = pred.loc[followup_visit.replace(".nii.gz", ""), "tla"]
        ntla_f_pred = pred.loc[followup_visit.replace(".nii.gz", ""), "ntla"]

        results.append([followup_visit.replace(".nii.gz", ""),
                        recist_outcome(tla_gt, tla_f_gt, ntla_gt, ntla_f_gt),
                        recist_outcome(tla_pred, tla_f_pred, ntla_pred, ntla_f_pred)])

results = pd.DataFrame(results)
results.columns = ["file", "gt", "pred"]
results.set_index("file", inplace=True)
ranked = sorted(results.index.tolist(), key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))
results_view = results.loc[ranked]

confusion_matrix(results_view["gt"].values, results_view["pred"].values)

