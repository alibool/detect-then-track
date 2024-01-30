# detect-then-track
This is the GitHub Repository providing an example code base for "Deep Learning-based Detect-then-track Pipeline for Treatment Outcome Assessments in Immunotherapy-treated Liver Cancer".

## Get started
The detect-then-track pipeline can be devided into two main parts, lesion detection and lesion tracking. The following subsection will provide a demo code for each step.

### Lesion detection
In lesion detection step, [nnDetection](https://github.com/MIC-DKFZ/nnDetection) were used. For comparative studies, Retina-Net and Faster-RCNN were adopted.

### Baseline-follow-up registration

Advanced Normalization Tools [ANTs](https://github.com/ANTsX/ANTsPy) were adopted. Besides, we compared the registration performances under three scenarios: without registration, non-deformable rigid registration, and deformable [DEEDS](https://github.com/mattiaspaul/deedsBCV) registration. 

