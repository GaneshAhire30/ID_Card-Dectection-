# ID_Card-Dectection-
# Business Case: ID Card Detection System


Businesses, organizations, and government agencies frequently handle identity documents such as Aadhaar cards, PAN cards, driving licenses, voter IDs, and employee ID cards.
Manually verifying or processing these documents is slow, error-prone, and not scalable.

This project provides an automated ID Card Detection system that can identify and locate ID-card-like objects in images, even under challenging conditions such as rotation, blur, partial occlusion, and noise.

# ID Card Detection using YOLOv5

A Deep Learning Project for Detecting Identity-Card-like Documents

This project implements a Document Detection Algorithm capable of detecting identity-card–like objects under various challenging conditions such as rotation, skew, noise, blur, and occlusion. The model is trained using YOLOv5 and a custom dataset created through Roboflow, and it outputs bounding boxes around detected ID cards.

This project fulfills the assignment requirements of creating a generalized algorithm for document boundary detection, including a detailed README, inference script, and dependency file.

<img width="555" height="228" alt="image" src="https://github.com/user-attachments/assets/51c16ae7-ed39-4b91-b9dc-21693f223cc2" />

  


## Problem Statement

Develop an algorithm that can:

* Detect boundaries of any identity card-like object

* Handle rotated, skewed, blurred, noisy, and partially occluded cards

* Perform well on a hidden validation set

* Use any model or framework

* Provide:

    * Complete README

    * Dependency file

    * Script to perform detection

This repository contains a full solution using YOLOv5.

## Dataset Preparation (Roboflow)

 <img width="1029" height="658" alt="image" src="https://github.com/user-attachments/assets/3fe756b8-23e0-404f-9a38-6953a551a087" />





Since no dataset was provided for the assignment, a custom dataset was created using images gathered from various reliable sources:

  * Kaggle: Publicly available ID card and document datasets

  * Google Images: Additional samples for diversity

  * Self-Captured Photos: Real-world ID card images taken using a mobile camera

All collected images were then uploaded to Roboflow, where they were labeled, processed, and augmented before being exported in YOLOv5 format.


* **Steps Performed:**

  * Uploaded raw ID card images from Kaggle, Google search, and self-captured photos

  * Manually labeled bounding boxes around ID cards

  * Applied extensive data augmentation within **Roboflow**

  * Exported final dataset in YOLOv5 format

* **Augmentations Used:**

  * Rotation

  * Noise

  * Perspective Transformation

  * Blur

  * Brightness/Contrast Adjustments

  * Flipping

   

These augmentations helped simulate various real-world conditions such as skew, tilt, low light, blur, and occlusion.


## Dataset Summary

The final dataset contains a single class: ID_Card.

* Total Images: 1,945
Dataset Split (Roboflow):

  * Training: 1,412 images

  * Validation: 391 images

  * Test: 142 images

This dataset is balanced, diverse, and robust, enabling the model to generalize well to different ID card appearances and environments.

* **Dataset Structure:**

dataset/
 ├── train/
 │    ├── images/
 │    └── labels/
 ├── val/
 │    ├── images/
 │    └── labels/
 ├── test/
 │    ├── images/
 └── data.yaml

* data.yaml Example:
{train: dataset/train/images
val: dataset/val/images
test: dataset/test/images

nc: 1
names: ['id_card']}

## Setup & Installation

Clone the YOLOv5 repository and install dependencies:
* Cloning the YoloV5 file from official repository.
* Changing the directory of yolov5
* Installing the dependencies
* Download all versions pre-trained weights


git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

## Model Choice: YOLOv5

YOLOv5 is chosen for its:

  * Fast training

  * Strong performance on document/object detection

  * Built-in augmentation

  * Ease of deployment

  * You experimented with multiple YOLOv5 model sizes:

    * YOLOv5s (small)

    * YOLOv5m (medium) — best performance
  * Higher model sizes such as m provided better generalization.

## Training

Training was performed using different image sizes and hyperparameters:
* Model 1: YOLOv5s (Image Size 128, Batch 8, Epochs 100)
* Model 2: YOLOv5s (Image Size 640, Batch 8, Epochs 100)
* Model 3: YOLOv5m (Image Size 128, Batch 8, Epochs 100)
* Model 4: YOLOv5m (Image Size 512, Batch 8, Epochs 100)
* Model 5: YOLOv5m (Image Size 640, Batch 16, Epochs 100)


## Training & Validation Metrics:-
YOLOv5 automatically generates training metrics including:

* Train Loss (Box, Objectness)

* Validation Loss

* Precision

* Recall

* F1 Score

* mAP@0.5

* mAP@0.5:0.95

* PR Curve

* Confusion Matrix

## AFTER TRAINING THE MODEL :-
* VISUALISE THE TRAINING METRICS

<img width="1489" height="1019" alt="image" src="https://github.com/user-attachments/assets/351a96fd-4fc9-40ce-ae99-56f5b08dc611" />


## Best Model Performance (YOLOv5m (Image Size 512, Batch 8, Epochs 100))

Precision: 0.995

Recall: 0.995

mAP@0.5: 0.995

mAP@0.5:0.95: 0.919

* Best mAP50-95 among all models -> 0.919

* Training and Validation scores are identical -> no overfitting

* Precision & Recall both = 0.995 -> extremely stable

* Higher resolution (512) improves small object + boundary detection

## PREDICTED TEST IMAGES:

<img width="851" height="1990" alt="image" src="https://github.com/user-attachments/assets/d8f621c0-9b74-4495-977b-551dc61f7782" />

## Tools and Technologies Learned

During this project, I gained hands-on experience with several tools and frameworks:

* YOLOv5: Implemented custom object detection models for ID cards.

* Roboflow: Learned to create, annotate, and manage custom datasets efficiently, including uploading images, labeling, and exporting in YOLO format.

* Python Libraries: OpenCV, Torch, NumPy, Matplotlib, and Seaborn for image processing, training, and visualization.

Learning Roboflow allowed me to streamline dataset preparation and ensured high-quality annotations for training my model.

## Conclusion :-
* The best performing model is YOLOv5m trained on 512×512 resolution, achieving a mAP50-95 of 0.919 on both training and validation datasets. The training and validation metrics are nearly identical, indicating no overfitting and excellent generalization.
* The YOLOv5m model delivered the best overall performance for ID card detection, achieving high precision and recall. It reliably detected identity-card-like objects even under challenging conditions such as rotation, skew, blur, noise, and partial occlusion. This model provides a robust baseline for real-world document-processing applications.
