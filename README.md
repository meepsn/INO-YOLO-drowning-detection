# INO-YOLO Drowning Detection

Independent research project, submitted to SARSEF in Spring 2026.

Won Second Place Grand Award, AVNET High School Innovator Award, Law and Policy

Abstract + Presentation Link: https://virtualfair.sarsef.org/2026/ino-yolo-a-lightweight-real-time-deep-learning-model-for-drowning-detection-on-low-cost-edge-devices/


Drowning Detection Deep-Learning Model using YOLO26 & with customized model architecture (BiFPN + LGCBlock). Inspired by YOLO11-LiB (Zhang et al. (2025)). 

Prioritized creating a lightweight model for lower-cost deployment of the system in low-income regions-- areas with the highest risk of unintentional drowning deaths.

Achieved DmAP50 of 0.90, Recall of 0.86, while having reduced computation cost being a 3 GFLOP, 2.95 MB, 1.4M parameter model. Additional benefit of being the highest FPS model due to lightweightedness.
___
# Dataset

Robust dataset: 2,000 original images of swimming/drowning, including images from various environments, lighting conditions, perspectives etc.

Data collected from Roboflow & other publically available sources. 

All images were labeled (drowning / swimming / background) in YOLO format through Roboflow annotation tool.

Data was split into traditional 7:2:1 ratio split. 
___

# Results

Comparative Expirimental Results

<img width="476" height="130" alt="image" src="https://github.com/user-attachments/assets/63957a87-2fbf-453d-91d7-2401cb0ca418" />

Training and Validation metrics over 100 Epochs

<img width="907" height="453" alt="image" src="https://github.com/user-attachments/assets/8950b44d-713d-4456-af9d-3f45fe8a19d0" />

___

# Limitations

Main limitation: over 70% of drowning deaths for teens aged 15-19 occur in open water

Solution: collect or simulate outdoor drowning/swimming dataset for training to increase model robustness

