import os
import pandas as pd
import numpy as np
import pickle
import SimpleITK as sitk
from tqdm import tqdm

status = "val"
pred_path = f"predictions/{status}_predictions/"
files = os.listdir(pred_path)

all_pred = []

for file in tqdm(files):
    livermask = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(livermask_path, file.replace("_boxes.pkl", "-livermask.nii"))))
    with open(os.path.join(pred_path, file), "rb") as f:
        pred_all = pickle.load(f)
        box = pred_all["pred_boxes"]
        pred_candidate = box[np.where(pred_all["pred_scores"] > 0.5)]
        for candidate in pred_candidate:
            if 1/2 < abs(int(candidate[1])-int(candidate[3])) / abs(int(candidate[4])-int(candidate[5])) < 2:
                all_pred.append([file.replace("_boxes.pkl", "")] + candidate.tolist())
            else:
                print("pop!")

save_pd = pd.DataFrame(all_pred)
save_pd.columns = ["file", "zmin", "xmin", "zmax", "xmax", "ymin", "ymax"]
save_pd.to_csv(f"pred_{status}.csv", index=False)
