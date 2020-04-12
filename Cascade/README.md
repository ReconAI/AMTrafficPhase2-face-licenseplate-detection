# Local Binary Pattern Cascade
Detection of faces and licene plates with a cascade classifier

Here I investigated, whether a cascade classifier would demonstrate needed performnce in speed and accuracy. This is a non-DL approach from classical CV, and would make a (strong) benchmark to our experiments. 

### License plate detection:
First I tried to train my own cascade, but that did not converge in reasonable time. Training a cascade requires a lot of empirical finetuning of parameters, and a complex model can require up to several days to train. Hence, I have resorted to using a trained cascade model from [OpenALPR](https://github.com/openalpr/openalpr) project. They have several cascade algorithms available for different regions, I have tested the eu.xml trained to recognize european license plates.

The cascade runs at ~70 ms on a system with OpenCV without CUDA support, and ~40 ms on a system with Opencvwith CUDA support.

Data to check speed/ visually assess performance:
- [EU number plates](https://drive.google.com/open?id=1aI0Ug1SqQgOZKWTDH8xr7996wUNCsW0I)

### HOW TO TEST

0. run ```pip3 install -r requirements.txt```
1. Download sample images and .xml 
2. Put the downloaded files in root folder
3. run ```python3 test_casade.py --in_folder {./eu} --out_path {path to save samples} --cascade_path eu.xml```


The script will save images with license plates highlighted, and print inference time to terminal.

## Face detection

There are a number of pre-trained cascade models available, most widely used ones coming with opencv library by default. However, they are not robust: one needs a separate model for frontal and lateral face detection. A quick test also demonstrated poor qualitative performance. Hence, I searched for other to other lightweight face detectors, see ../Lightweight_face_detector.

