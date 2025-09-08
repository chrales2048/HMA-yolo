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

The architecture balances **high accuracy** with **real-time performance**, achieving high FPS with significant accuracy improvements.
---
## ⚙️ Repository Structure
HMA-YOLO/
│-- models/          # Core modules
│-- requirements.txt # Dependencies
│-- subbranch_removal.py         # 
│-- train.py          # Minimal demo script
│-- val.py          # Minimal demo script
│-- detect.py          # Minimal demo script
│-- README.md        # Project documentation
 ## 📖 Citation
 If you find this work useful, please cite our manuscript:
 @article{Peng2025HMA-YOLO,
  title   = {Enhanceing Gangue Recognition in Coal Mines：A Lightweight Network with Multi-Path Attention},
  author  = {Zheng Wang and Le Pengu and Yujiang Liu},
  journal = {The Visual Computer},
  year    = {2025}
}
🔗 Related Information
This repository is directly associated with the manuscript submitted to The Visual Computer.
