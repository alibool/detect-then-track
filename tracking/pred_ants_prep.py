import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

status = "Ts"
image_path = f"/ants/images{status}/"
annot = pd.read_csv(f"pred_{status.lower()}.csv")
save_mask_path = f"predictions/pred_masks{status}_prep/"
os.makedirs(save_mask_path, exist_ok=True)

files = annot.loc[:, "file"].tolist()
annot.set_index("file", inplace=True)

for file in tqdm(list(set(files))):
    img = sitk.ReadImage(os.path.join(image_path, f"{file}_0000.nii.gz"))
    mask = np.zeros_like(sitk.GetArrayFromImage(img))
    annot_info = annot.loc[[file]]
    for i in range(annot_info.shape[0]):
        info = annot_info.iloc[i]
        x_center = (info["xmin"] + info["xmax"]) // 2
        y_center = (info["ymin"] + info["ymax"]) // 2
        z_center = (info["zmin"] + info["zmax"]) // 2
        mask[int(z_center-5):int(z_center+5),
             int(x_center-5):int(x_center+5),
             int(y_center-5):int(y_center+5)] = int(i+1)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetOrigin(img.GetOrigin())
    mask_img.SetSpacing(img.GetSpacing())
    mask_img.SetDirection(img.GetDirection())
    sitk.WriteImage(mask_img, os.path.join(save_mask_path, f"{file}.nii.gz"))
