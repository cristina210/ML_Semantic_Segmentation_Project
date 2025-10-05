## Overview
This project investigates semantic segmentation using deep convolutional neural networks, focusing on the challenges posed
by the domain gap between synthetic and real-world datasets.
Training semantic segmentation models typically requires large-scale annotated datasets, which are expensive 
and time-consuming to produce. Synthetic datasets, such as GTA5, offer a cost-effective alternative,
but models trained on synthetic data often underperform on real images like those in Cityscapes.
We evaluate two representative architectures, DeepLabV2 and BiSeNet, on both synthetic and real datasets, 
analyzing performance metrics including mean Intersection-over-Union (mIoU). Furthermore, we explore data augmentation and 
an adversarial domain adaptation techniques to improve generalization across domains with Bisenet architecture.

## Dataset
Cityscapes: Real-world urban scene images with pixel-level annotations.
https://www.cityscapes-dataset.com/

GTA5: Synthetic dataset generated from the Grand Theft Auto V game.

## Architectures
**BiSeNet** architecture for real-time semantic segmentation with a pre-trained **ResNet18** backbone.
**DeepLabV2**  architecture with a pre-trained **ResNet101** backbone.

## Loss functions and optimizer
Loss functions:
- Segmentation Loss: **CrossEntropyLoss** 
- Adversarial Loss: **BCEWithLogitsLoss** for domain adaptation

Optimizers:
- Segmentation model: **SGD** with momentum and weight_decay
- Discriminator (adversarial): **Adam optimizer** 

Optional extension:
- Use Adam optimizer for the segmentation model 
- Experiment with Focal Loss as an alternative to CrossEntropy

## Goal
step2a and step2b: Comparison between two architectures, DeepLabV2 and BiSeNet in terms of accuracy and computational times (FLOP and latency).
step3a3b: Analysis of Bisenet performance in domain shift and usage of data augmentation
step4_5:  
- Analysis of Bisenet performance in domain shift with an adversarial domain adaptation approach
- Studying the effect of using as loss function for the segmentation model a Focal loss and as optimizer Adam optimizer.

# Main libraries
numpy, matplotlib, Pillow
# PyTorch
torch, torchvision

