import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
from datetime import datetime
from tqdm import tqdm

status = "Ts"
before_register = f"pred_{status.lower()}.csv"
before = pd.read_csv(before_register)
mask_path = f"masks{status}_ants/"
image_path = f"images{status}/"
baseline_path = f"pred_masks{status}_prep/"

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
            all_lesions.append([baseline, i, x_b, y_b, z_b, diameter])
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
                all_lesions.append([followup_visit, i, x_f, y_f, z_f, diameter])

save_pd = pd.DataFrame(all_lesions)
save_pd.columns = ["file", "idx", "x", "y", "z", "diameter"]
# save_pd.to_csv(f"pred_ants_{status.lower()}.csv", index=False)


def calculate_iou_3d(box1, box2):

    # x1_box1, y1_box1, z1_box1, x2_box1, y2_box1, z2_box1 = box1
    # x1_box2, y1_box2, z1_box2, x2_box2, y2_box2, z2_box2 = box2

    z1_box1, x1_box1, z2_box1, x2_box1, y1_box1, y2_box1 = box1
    z1_box2, x1_box2, z2_box2, x2_box2, y1_box2, y2_box2 = box2

    volume_box1 = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1) * (z2_box1 - z1_box1 + 1)
    volume_box2 = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1) * (z2_box2 - z1_box2 + 1)

    x_left = max(x1_box1, x1_box2)
    y_top = max(y1_box1, y1_box2)
    z_front = max(z1_box1, z1_box2)
    x_right = min(x2_box1, x2_box2)
    y_bottom = min(y2_box1, y2_box2)
    z_back = min(z2_box1, z2_box2)

    intersection_volume = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1) * max(0, z_back - z_front + 1)
    union_volume = volume_box1 + volume_box2 - intersection_volume
    iou = intersection_volume / union_volume

    return iou


def calculate_center(box):
    z1_box1, x1_box1, z2_box1, x2_box1, y1_box1, y2_box1 = box
    return np.array([(x1_box1 + x2_box1) / 2, (y1_box1 + y2_box1) / 2, (z1_box1 + z2_box1) / 2])


img_path = f"images{status}/"

files = os.listdir(img_path)
files = [x.replace("_0000", "") for x in files]

if status == "Val":
    patients = list(set([x[:9] for x in files]))
else:
    patients = list(set([x[:8] for x in files]))

gt = pd.read_csv(f"annot_target_{status.lower()}.csv")
gt.set_index("file", inplace=True)

pred = save_pd
pred.iloc[:, 0] = pred.iloc[:, 0].apply(lambda x: x.replace("_0000.nii.gz", ""))
pred.set_index("file", inplace=True)

all_ious = []

for patient in tqdm(patients):
    patient_visits = list(filter(lambda x: x.startswith(patient), files))
    if status == "Ts":
        baseline = min(patient_visits, key=lambda visit: datetime.strptime(visit[9:17], "%Y%m%d"))
    else:
        baseline = min(patient_visits, key=lambda visit: int(visit.split(".")[0].split("-")[1]))

    patient_visits.remove(baseline)

    if baseline.replace(".nii.gz", "") in pred.index.tolist():
        baseline_info = pred.loc[[baseline.replace(".nii.gz", "")]]
        baseline_info.sort_values("diameter", ascending=False, inplace=True)
        candidate_tlas = baseline_info[baseline_info["diameter"] >= 10].iloc[:2]

        origin_b = before.loc[[baseline.replace(".nii.gz", "")]]
        origin_b_idx = candidate_tlas["idx"]
        if baseline.replace(".nii.gz", "") in gt.index:
            print(f"Baseline{baseline}")
            gt_b = gt.loc[[baseline.replace(".nii.gz", "")]]
            gt_b.set_index("tag", inplace=True)
            # 1/2 target lesion
            for b_idx in origin_b_idx:
                pred_origin_b = origin_b.iloc[b_idx]
                iou_tmp = []
                bbox_tmp = []
                for gt_tmp in range(gt_b.shape[0]):
                    gt_box = gt_b.iloc[gt_tmp][["z1", "y1", "z2", "y2", "x1", "x2"]]

                    center = [np.mean(np.array([gt_box["x1"], gt_box["x2"]])),
                              np.mean(np.array([gt_box["y1"], gt_box["y2"]]))]
                    rad = np.sqrt((gt_box["x1"] - center[0]) ** 2 + (gt_box["y1"] - center[1]) ** 2)

                    gt_final_box_b = np.array([gt_box.values[0], center[1] - rad,
                                               gt_box.values[2], center[1] + rad,
                                               center[0] - rad, center[0] + rad])
                    bbox_tmp.append(gt_final_box_b)
                    iou_tmp.append(calculate_iou_3d(pred_origin_b.values, gt_final_box_b))
                tag_select = gt_b.iloc[[np.argmax(iou_tmp)]].index

                deviation_f = abs(calculate_center(pred_origin_b.values) -
                                  calculate_center(bbox_tmp[np.argmax(iou_tmp)]))

                all_ious.append([baseline, np.max(iou_tmp), deviation_f[0] * 0.8,
                                 deviation_f[1] * 0.8, deviation_f[2] * 5])

                for followup_visit in patient_visits:
                    if followup_visit.replace(".nii.gz", "") in pred.index:
                        followup_info = pred.loc[[followup_visit.replace(".nii.gz", "")]]
                        origin_f = before.loc[[followup_visit.replace(".nii.gz", "")]]

                        gt_f = gt.loc[[followup_visit.replace(".nii.gz", "")]]
                        gt_f.set_index("tag", inplace=True)

                        if tag_select.values[0] in gt_f.index:

                            gt_f_select = gt_f.loc[tag_select, ["z1", "y1", "z2", "y2", "x1", "x2"]]

                            loc_b = candidate_tlas[candidate_tlas["idx"] == b_idx].loc[:, ["x", "y", "z"]].values
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
                                pred_origin_f = origin_f.iloc[int(followup_info.iloc[num]["idx"])]

                                center = [np.mean(np.array([gt_f_select["x1"], gt_f_select["x2"]])),
                                          np.mean(np.array([gt_f_select["y1"], gt_f_select["y2"]]))]

                                rad = np.sqrt((gt_f_select["x1"].values[0] - center[0]) ** 2 +
                                              (gt_f_select["y1"].values[0] - center[1]) ** 2)

                                gt_final_box_f = np.array([gt_f_select.values[0, 0], center[1] - rad,
                                                           gt_f_select.values[0, 2], center[1] + rad,
                                                           center[0] - rad, center[0] + rad])

                                deviation_f = abs(calculate_center(gt_final_box_f) - calculate_center(pred_origin_f))

                                all_ious.append([followup_visit, calculate_iou_3d(pred_origin_f.values, gt_final_box_f),
                                                 deviation_f[0] * 0.8, deviation_f[1] * 0.8, deviation_f[2] * 5])

save_all = pd.DataFrame(all_ious)
save_all.columns = ["file", "iou", "x", "y", "z"]
save_all.to_csv(f"save_iou_{status}.csv", index=False)
