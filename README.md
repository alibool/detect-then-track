# detect-then-track
This is the GitHub Repository providing an example code base for "Deep Learning-based Detect-then-track Pipeline for Treatment Outcome Assessments in Immunotherapy-treated Liver Cancer".

![model](document/workflow.png)

## Get started
The detect-then-track pipeline can be devided into two main parts, lesion detection and lesion tracking. The following subsection will provide a demo code for each step.

### Lesion detection
In lesion detection step, [nnDetection](https://github.com/MIC-DKFZ/nnDetection) were used. nnDetection is based on Retina-Unet. Weakly supervision of masks are needed. In this work, a 2D slice with the largest tumor diameter in axial view was selected from the 3D scan for each liver lesion, and annotated the two endpoints of the diameter. The annotated diameter was used to generate the pseudo-mask.

Run `python detect_prep/pseudomask.py` for pseudo-mask generation. Inputs are defined below.

|  input  | description |
| ------------------- | ------------- |
| `img_base`  | base directory for the input CT scans |
| `save_base` | saving directory for the generated pseudo-mask |
| `csv_path`  | a xml file of annotated two endpoints of the diameter |

For comparative studies, Retina-Net and Faster-RCNN were adopted.

### Baseline-follow-up registration

Advanced Normalization Tools [(ANTs)](https://github.com/ANTsX/ANTsPy) were adopted. Besides, we compared the registration performances under three scenarios: without registration, non-deformable rigid registration, and deformable [DEEDS](https://github.com/mattiaspaul/deedsBCV) registration. 

### 
