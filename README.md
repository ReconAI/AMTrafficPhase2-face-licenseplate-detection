# AMTrafficPhase2-face-licenseplate-detection

Task # 2 from Specifications

Prepare and develop end-to-end pipeline (from dataset aggregation to
network deployment on end device) for license plate and face detection
light-weight neural network.
Project must contain a comparison of following approaches:
1. Detection using miscellaneous Haar-cascade classifier (Viola-Jones
detector) implementations (OpenCV, Dlib and other open source)
2. Detection using Nvidia DetectNet (Resnet-10,18 etc.). DIGITS training
and Nvidia TensorRT convertor detection can be considered as an
approach.
3. Other approaches found during a research

I used the following approaches for the project:
1. [Cascade classifier](https://github.com/ReconAI/AMTrafficPhase2-face-licenseplate-detection/tree/master/Cascade)
2. [Lightweight face detector](https://github.com/ReconAI/AMTrafficPhase2-face-licenseplate-detection/tree/master/Lightweight_face%20_detector)
3. [Detectnet_v2](https://github.com/ReconAI/AMTrafficPhase2-face-licenseplate-detection/tree/master/Detectnet_resnet10)
4. [Retinanet](https://github.com/ReconAI/AMTrafficPhase2-face-licenseplate-detection/tree/master/Retinanet)

example testing images for face and license plate detection are also included, for consistency of evaluation across methods.

Detailed specification can be found in each subproject's README

The final submitted solution was built using Detectnet based on Resnet10, demonstrating 28 FPS on Jetson Nano and 0.7 mAP on validation set.


