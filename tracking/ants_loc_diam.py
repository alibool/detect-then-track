import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
from datetime import datetime
from tqdm import tqdm

status = "Ts"
before_register = f"pred_{status.lower()}.csv"
before = pd.read_csv(before_register)
mask_path = f"predictions/masks{status}_ants/"
image_path = f"ants/images{status}/"
baseline_path = f"predictions/pred_masks{status}_prep/"

files = before.loc[:, "file"].tolist()
before.set_index("file", inplace=True)

if status == "Ts":
    patients = list(set([x[:8] for x in os.listdir(image_path)]))
else:
    patients = list(set([x[:9] for x in os.listdir(image_path)]))
files = os.listdir(image_path)


def return_diameter(info):
    x = (info["xmin"] + info["xmax"]) // 2
    y = (info["ymin"] + info["ymax"]) // 2
    z = (info["zmin"] + info["zmax"]) // 2
    d = max(abs(info["xmax"] - info["xmin"]), abs(info["ymax"] - info["ymin"]))
    return x, y, z, d


all_lesions = []
for patient in tqdm(sorted(patients)):
    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    if status == "Ts":
        baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[9:17], "%Y%m%d"))
    else:
        baseline = min(patient_visits, key=lambda visit: int(visit.split(".")[0].split("-")[1]))
    patient_visits.remove(baseline)
    if baseline.replace("_0000.nii.gz", "") in before.index.tolist():
        ants_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(baseline_path, baseline.replace("_0000", ""))))
        info_baseline = before.loc[[baseline.replace("_0000.nii.gz", "")]]
        for i in range(info_baseline.shape[0]):
            info_b = info_baseline.iloc[i]
            z_b = np.mean(np.where(ants_mask == i + 1)[0])
            y_b = np.mean(np.where(ants_mask == i + 1)[1])
            x_b = np.mean(np.where(ants_mask == i + 1)[2])
            _, _, _, diameter = return_diameter(info_b)
            all_lesions.append([baseline, x_b, y_b, z_b, diameter])
    for followup_visit in patient_visits:
        if followup_visit.replace("_0000.nii.gz", "") in before.index.tolist():
            info_followup = before.loc[[followup_visit.replace("_0000.nii.gz", "")]]
            ants_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, followup_visit.replace("_0000", ""))))
            for i in range(info_followup.shape[0]):
                info_f = info_followup.iloc[i]
                z_f = np.mean(np.where(ants_mask == i+1)[0])
                y_f = np.mean(np.where(ants_mask == i+1)[1])
                x_f = np.mean(np.where(ants_mask == i+1)[2])
                _, _, _, diameter = return_diameter(info_f)
                all_lesions.append([followup_visit, x_f, y_f, z_f, diameter])

save_pd = pd.DataFrame(all_lesions)
save_pd.columns = ["file", "x", "y", "z", "diameter"]
save_pd.to_csv(f"pred_ants_{status.lower()}.csv", index=False)
