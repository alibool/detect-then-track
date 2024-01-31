import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np

status = "Ts"
img_path = f"images{status}/"
files = os.listdir(img_path)
files = [x.replace("_0000", "") for x in files]

if status == "Val":
    patients = list(set([x[:9] for x in files]))
else:
    patients = list(set([x[:8] for x in files]))

pred = pd.read_csv(f"pred_ants_{status.lower()}.csv")

pred.iloc[:, 0] = pred.iloc[:, 0].apply(lambda x: x.replace("_0000.nii.gz", ""))
pred.set_index("file", inplace=True)

pred_record = []
for patient in tqdm(patients):
    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    if status == "Ts":
        baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[9:17], "%Y%m%d"))
    else:
        baseline = min(patient_visits, key=lambda visit: int(visit.split(".")[0].split("-")[1]))

    select_num = 0
    tla_diameter, ntla_diameter = 0, 0
    if baseline.replace(".nii.gz", "") in pred.index.tolist():
        baseline_info = pred.loc[[baseline.replace(".nii.gz", "")]]
        baseline_info.sort_values("diameter", ascending=False, inplace=True)
        candidate_tlas = baseline_info[baseline_info["diameter"] >= 10].iloc[:2]
        tla_diameter = candidate_tlas["diameter"].values.sum()
        ntla_diameter = baseline_info["diameter"].values.sum() - tla_diameter
        select_num = candidate_tlas.shape[0]
    pred_record.append([baseline.replace(".nii.gz", ""), tla_diameter, ntla_diameter, select_num])
    patient_visits.remove(baseline)
    for followup_visit in patient_visits:
        tla_diameter_f, ntla_diameter_f = 0, 0
        if followup_visit.replace(".nii.gz", "") in pred.index.tolist():
            followup_info = pred.loc[[followup_visit.replace(".nii.gz", "")]]
            ntla_diameter_f = followup_info["diameter"].values.sum()
            if select_num:
                for i in range(select_num):
                    loc_b = candidate_tlas.loc[:, ["x", "y", "z"]].iloc[i, :].values
                    loc_f = followup_info.loc[:, ["x", "y", "z"]].values
                    distance = np.sqrt((((loc_f - loc_b) * np.array([0.8, 0.8, 5])) ** 2).sum(1))
                    candidates = np.where(distance < 30)[0]
                    if candidates.shape[0] == 1:
                        num = candidates[0]
                    elif candidates.shape[0] > 1:
                        num = np.argmin(distance)
                    else:
                        num = 999
                    if num < 100:
                        temp_diam = float(followup_info.iloc[num, :].loc["diameter"])
                        tla_diameter_f = tla_diameter_f + temp_diam
                        followup_info.reset_index(inplace=True)
                        followup_info.drop(num)
                        followup_info.set_index("file", inplace=True)
            ntla_diameter_f = ntla_diameter_f - tla_diameter_f
            assert ntla_diameter_f >= 0
        pred_record.append([followup_visit.replace(".nii.gz", ""), tla_diameter_f, ntla_diameter_f, select_num])

save_pd = pd.DataFrame(pred_record)
save_pd.columns = ["file", "tla", "ntla", "target_num"]
save_pd.to_csv(f"pred_tla_ntla_{status.lower()}.csv", index=False)
