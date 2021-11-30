# GCOOD: A Generic Coupled Out-of-Distribution Detector for Robust Classification

## Abstract
Neural networks have achieved high degrees of accuracy in classification tasks. However, when an out-of-distribution (OOD) sample (\emph{i.e.,}~entries from unknown classes) is submitted to the classification process, the result is the association of the sample to one or more of the trained classes with different degrees of confidence. If any of these confidence values are more significant than the user-defined threshold, the network will mislabel the sample, affecting the model credibility. The definition of the acceptance threshold itself is a sensitive issue in the face of the classifier's overconfidence. This paper presents the Generic Coupled OOD Detector (GCOOD), a novel Convolutional Neural Network (CNN) tailored to detect whether an entry submitted to a trained classification model is an OOD sample for that model. From the analysis of the Softmax output of any classifier, our approach can indicate whether the resulting classification should be considered or not as a sample of some of the trained classes. To train our CNN, we had to develop a novel training strategy based on Voronoi diagrams of the location of representative entries in the latent space of the classification model and graph coloring. We evaluated our approach using ResNet, VGG, DenseNet, and SqueezeNet classifiers with images from the CIFAR-10 dataset.


## The versions of programs used were
- Anaconda 3.7.4
- PyTorch 1.7.1
- Albumentations 0.5.2
** Installing Albumentations 0.5.2 can be done using the command pip install albumentations==0.5.2**

## The datasets used
The datasets used were CIFAR10 and CIFAR100.

## Download pre-trained model
The data folder has several subfolders, but they are all empty. To download this folder filled with your files, download it from the link https://drive.google.com/drive/folders/1OEbSVyw25FqD7Q2itIH39PI0Lf5bXMF- and replace the CIFAR10 in-data folder with this one.





