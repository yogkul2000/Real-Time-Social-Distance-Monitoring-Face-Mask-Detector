# Real-Time-Social-Distance-Monitoring-Face-Mask-Detector

### This project leverages Computer Vision to monitor social distancing at any place and also has ability to track people who are wearing masks.


### Samples

 ![](val.gif)
 ![](test.gif)

### Usage
Just download the following files and place them in the YOLO folder:
1. YOLOv3 spp weights :  https://pjreddie.com/media/files/yolov3-spp.weights
2. Face Mask Classifier ResNet50 Keras Model : https://drive.google.com/file/d/1nPJzjzUt9qqpLAONtKXjfYNXUWlPhBxU/view?usp=sharing


## Description and Results
- Utilized YOLOv3 fordetecting persons fromreal-time CCTV feed, a Dual Shot Face Detector for detecting faces composed of layers from the VGG16 network and a ResNet model for performing Face Mask classification.
- The Proposed Framework achieved an accuracy of 84.3% with an average of 44 fps on NVIDIA GTX 1660 Ti.
- The goal of this project was to come up with an accurate prediction pipeline for real – time social distance and facemask violation during these tough times of pandemic.


