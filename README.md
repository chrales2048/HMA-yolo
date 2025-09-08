# HMA-yolo
This repository contains the official implementation of the paper:  

> *Enhanceing Gangue Recognition in Coal Mines：A Lightweight Network with Multi-Path Attention**  
> Submitted to *The Visual Computer* (2025).
-------
 ## 📌 Overview
HMA-YOLO is a lightweight deep learning framework designed for robust object detection in challenging environments such as coal preparation plants.  
Key innovations include:
- **HLAnet**: Low-light enhancement and motion-blur correction  
- **Multi-path Large Kernel Attention (MLKA)**: Parallel attention branches for capturing local textures and global contours  
- **Adaptive Frequency Decomposition Module (AFDM)**: Improved boundary-aware feature fusion  
- **Ghost-InceptionV2 (GI Conv)**: Efficient convolution with reduced redundancy  

The proposed architecture achieves a balanced trade-off between **accuracy** and **real-time performance**, showing significant improvements in gangue detection tasks
---
## ⚙️ Repository Structure
HMA-YOLO/
│-- models/          # Core modules
│-- requirements.txt # List of dependencies for environment setup
│-- yolov9-s-converted.pt         #  YOLOv9 weight file
│-- subbranch_removal.py         # Utility script for pruning/removing redundant branches (structural simplification)
│-- train.py          # Training script
│-- val.py          # Validation script
│-- detect.py          # Inference demo script 
│-- README.md        # Project documentation
 ## 📖 Citation
 If you find this work useful, please cite our manuscript:
 @article{Peng2025HMA-YOLO,
  title   = {Enhanceing Gangue Recognition in Coal Mines：A Lightweight Network with Multi-Path Attention},
  author  = {Zheng Wang and Le Peng and Yujiang Liu},
  journal = {The Visual Computer},
  year    = {2025}
}
 ##🔗 Related Information
This repository is directly associated with the manuscript submitted to The Visual Computer.
##🙏Acknowledgements
https://github.com/WongKinYiu/yolov7
