import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm


def writeJson(path, count):
    with open(path, "w+", encoding="utf-8") as f:
        f.write('{\n')
        f.write('\t"instances":{\n')
        for i in range(1, count-1):
            f.write('\t \t "%s":0, \n' % i)
        if count > 1:
            f.write('\t \t "%s":0 \n' % str(count-1))
        f.write('\t }\n')
        f.write('}')


img_base = "imagesTs/"
save_base = "labelsTs/"
csv_path = "annot_ts.csv"
annot_file = pd.read_csv(csv_path, index_col=0)
files = list(set(annot_file.index.tolist()))

for file in tqdm(files):
    print(file)
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_base, f"{file}_0000.nii.gz")))
    pseudo_mask = np.zeros_like(img)
    box_annotations = annot_file.loc[[file]]
    count = 1
    for i in range(box_annotations.shape[0]):
        single_annot = box_annotations.iloc[i, :]
        center = [np.mean(np.array([single_annot["x1"], single_annot["x2"]])),
                  np.mean(np.array([single_annot["y1"], single_annot["y2"]]))]
        rad = np.sqrt((single_annot["x1"]-center[0])**2 + (single_annot["y1"]-center[1])**2)
        print(rad)
        for y in range(int(center[1] - rad - 2), int(center[1] + rad + 2)):
            for x in range(int(center[0] - rad - 2), int(center[0] + rad + 2)):
                if (y - center[1]) ** 2 + (x - center[0]) ** 2 <= rad ** 2:
                    p_min = min(single_annot["z1"], single_annot["z2"])
                    p_max = max(single_annot["z1"], single_annot["z2"])
                    pseudo_mask[int(p_min):int(p_max) + 1, y, x] = count
        count = count + 1

    save_mask = sitk.GetImageFromArray(pseudo_mask)
    save_seg_path = os.path.join(save_base, file.split('.')[0] + '.nii.gz')
    sitk.WriteImage(save_mask, save_seg_path)
    save_json = os.path.join(save_base, file.split('.')[0] + '.json')
    writeJson(save_json, count)

