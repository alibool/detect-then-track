import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np

status = "Val"
img_path = f"images{status}/"
files = os.listdir(img_path)

if status == "Val":
    patients = list(set([x[:9] for x in files]))
else:
    patients = list(set([x[:8] for x in files]))

base_path = "predictions_final/"
gt = pd.read_csv(os.path.join(base_path, f"annot_{status.lower()}.csv"))
gt.set_index("file", inplace=True)


def return_diameter(info):
    x = (info["x1"] + info["x2"]) // 2
    y = (info["y1"] + info["y2"]) // 2
    z = (info["z1"] + info["z2"]) // 2
    d = np.sqrt((info["x1"]-info["x2"])**2 + (info["y1"]-info["y2"])**2)
    return x, y, z, d


gt_record = []
for patient in tqdm(patients):
    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    if status == "Ts":
        baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[9:17], "%Y%m%d"))
    else:
        baseline = min(patient_visits, key=lambda visit: int(visit.split(".")[0].split("-")[1]))
    # baseline gt diameter calculation
    select_num = 0
    gt_tla_diameter, gt_ntla_diameter = 0, 0
    if baseline.replace("_0000.nii.gz", "") in gt.index.tolist():
        baseline_info = gt.loc[[baseline.replace("_0000.nii.gz", "")]]
        tags = baseline_info.loc[:, "tag"].tolist()
        if "liver-1" in tags:
            tla_info_1 = baseline_info.loc[baseline_info["tag"] == "liver-1"]
            x_1, y_1, z_1, d_1 = return_diameter(tla_info_1)
            gt_tla_diameter = float(d_1)
            select_num += 1
        if "liver-2" in tags:
            tla_info_2 = baseline_info.loc[baseline_info["tag"] == "liver-2"]
            x_2, y_2, z_2, d_2 = return_diameter(tla_info_2)
            gt_tla_diameter = gt_tla_diameter + float(d_2)
            select_num += 1
        if "NL" in tags:
            ntla_info = baseline_info.loc[baseline_info["tag"] == "NL"]
            for i in range(ntla_info.shape[0]):
                ntla_info_single = ntla_info.iloc[i]
                _, _, _, d_ntla = return_diameter(ntla_info_single)
                gt_ntla_diameter = gt_ntla_diameter + float(d_ntla)
    gt_record.append([baseline.replace("_0000.nii.gz", ""), gt_tla_diameter, gt_ntla_diameter, select_num])
    patient_visits.remove(baseline)
    for followup_visit in patient_visits:
        select_num_f = 0
        gt_tla_diameter_f, gt_ntla_diameter_f = 0, 0
        if followup_visit.replace("_0000.nii.gz", "") in gt.index.tolist():
            followup_info = gt.loc[[followup_visit.replace("_0000.nii.gz", "")]]
            tags = followup_info.loc[:, "tag"].tolist()
            if "liver-1" in tags:
                tla_info_1 = followup_info.loc[followup_info["tag"] == "liver-1"]
                x_1, y_1, z_1, d_1 = return_diameter(tla_info_1)
                gt_tla_diameter_f = float(d_1)
                select_num_f += 1
            if "liver-2" in tags:
                tla_info_2 = followup_info.loc[followup_info["tag"] == "liver-2"]
                x_2, y_2, z_2, d_2 = return_diameter(tla_info_2)
                gt_tla_diameter_f = gt_tla_diameter_f + float(d_2)
                select_num_f += 1
            if "NL" in tags:
                ntla_info = followup_info.loc[followup_info["tag"] == "NL"]
                for i in range(ntla_info.shape[0]):
                    ntla_info_single = ntla_info.iloc[i]
                    _, _, _, d_ntla = return_diameter(ntla_info_single)
                    gt_ntla_diameter_f = gt_ntla_diameter_f + float(d_ntla)
        gt_record.append([followup_visit.replace("_0000.nii.gz", ""), gt_tla_diameter_f, gt_ntla_diameter_f, select_num_f])

save_pd = pd.DataFrame(gt_record)
save_pd.columns = ["file", "tla", "ntla", "target_num"]
save_pd.to_csv(f"gt_tla_ntla_{status.lower()}.csv", index=False)
