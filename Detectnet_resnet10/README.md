## Detectnet_v2 based on resnet-10
[Detectnet](https://devblogs.nvidia.com/detectnet-deep-neural-network-object-detection-digits/) is a network architecture developed by NVIDIA as a single-shot lightweight object detector. It is based on a sliding window approach: the inital image is split up into a grid of cells, and for each cell, network predicts whether it is occupied with an object and if so, this object's bounding box. It is used in most of Nvidia's sample applications, and is rather robust in terms of backbone architectures accepted (resnets, mobilenets). Nvidia provides tools to train such models and get them in format that Deepstream apps can digest. As we want our final model to be integrated into Deepstream app, and want it to be optimized for running on Nano, I decided to go for Nvidia's own DL tools. There are 2 Nvidia's products for training networks optimized for edge devices: [DIGITS](https://developer.nvidia.com/digits) and [TLT](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#gettingstarted_overview). Of the two, TLT is a newer product and is more in-line with modern DL tools (uses tensorflow for training, tfrecords for data routines and tensorboard for logging). DIGITS runs on Caffe, has rather obscure configuration files for network architectures, proprietary data storage format and visualization interfaces. Also, TLT comes with ready-made utilities for model compression and conversion to a format expected by Deepstream, as well as a number of other architectures (classifiacation models, SSD, Fast-Rcnn). Having summed up pros and cons, I decided to go with TLT. 

In TLT, there are several options for a detector: Detectnet, SSD or Fast-Rcnn. However, only Detectnet is currently integrated easily into Deepstream apps. For other two, one needs to prepare a C++ module to parse the output of the network. Also, most of the examples by NVIDIA about their ultra fast solutions runnin on Nano were based on Detectnet architecture. This is why I have chosen it for the experiments.

First I trained a resnet18 based model. This one had two drawbacks- first, it did not detect smaller license plates because the dataset contained mostly car shots from near range. Second, it was running only 22 FPS max. Having these results, I decided to go for a lighter resnet10 backbone with more extensive scaling augmentations- i.e. zooming out the image to make LPs smaller. This resulted in a more robust model with maximum speed of **28 FPS (35 ms)** after all optimizations. The below submission (weights and configs and results) represents this model.

### Framework deployment
TLT comes as NVIDIA docker container. [The official documentation](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#requirements) on the steps to install it is provided. It is recommended to run it with mounting local directories for data and outputs, in order to have results saved after container shutdown.

After installation, download the pre-trained models as per [here](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#downloading_models). The desired model is resnet10-based detectnet_v2.

### Dataset preparation
Data must come in KITTI format. The dataset I assembled from multiple sources [here](https://drive.google.com/open?id=1JCnEdYGF9HPjbH5lEVhFBW3_3AuiiN8n) is already in this format. The data preprocessing required includes
- scaling the images to 640 px on longer side
- padding shorter sides to reach 640x640 resolution

Next, data must be converted to TFRecords as in [conversion guide](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#conv_tfrecords_topic). THe config file to provide as <path_to_tfrecords_conversion_spec> is tlt_dataconvert_config.cfg

### Training commands
For training, launch the [tlt-train utility](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#training_gridbox). 

an experiment configuration file (`--spec_file`) is to be submitted, use experiment_config.cfg or refer to [docs](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#create_exp_spec_file_topic) for different config values.

training logs will be saved to `results_dir` specified when launching training command, use tensorboard to view them. Model checkpoints are also saved to this directory, in .tlt format, specific to TLT framework.

After training, the model can be pruned- that is, the layers and features that are not useful for predicting correct lables will be removed from model. This results in a model with similar performance but of only ~15% of original size. refer to [docs](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#pruning_models) for instructions. Default values seem to work well. 

After pruning, official documentation recommends re-training the pruned model. However, I did not notice improvements for re-trained pruned model over simply pruned one. Refer to docs for re-training and prunned_config.cfg for training specification file.

### Testing commands

To evaluate model on the testing set, follow these steps:
- Follow instructions above to set up the TLT environment, mounting a host directory to the container ('data' in the following example). Enter the container, the sample command is
`sudo docker run --runtime=nvidia -it -v /media/hdd/data:/workspace/data 
--rm -p 8228:8888 nvcr.io/nvidia/tlt-streamanalytics:v1.0_py2`

- Unzip the following [files](https://drive.google.com/open?id=1ivHds9H-JwgrF98dHvx9JePwaX29oS1B) to the directory you mounted (e.g. 'data'). Keep directory structure (not creating extra folders). In 'data' there should be now a .cfg file and a 'test' folder with images subfolder and a tfrecords file.
- Download the .tlt file from the 'Weight links' section below. Also put it to the mounted directory (e.g. 'data')
- Run the following command
```
tlt-evaluate detectnet_v2 -e data/testing_config.cfg  -m data/model.step-72920.tlt  -k flp
```
This will prepare the evaluation and run it, reporting mAP (~86%) and per-class APs. 

#### Image visualization inference

Inference on the training host is done via [tlt-infer utility](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#inference_detectnet_v2). For this, postprocessing config (`--cluster_params_file`) must be submitted. In this file we define parameters to cluster near-by bounding boxes into single object (instead of NMS used in other frameworks), specify confidence hresholds, etc. In this project, this file is ./cluster_params.json.

### Deployment on Nano
When the model is ready for export, we can export it to a format Deepstream apps can ingest (.etlt file). After receiving it for the first time, Deepstream will automatically create and save a TensorRT .engine file with machine-specific optimizations, which can be used afterwards directly. Again, refer to [docs](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html#exporting_models) for extensive info.

The command for exporting a pruned resnet10-detectnet_v2 in FP16 mode was as follows
```
tlt-export --export_module detectnet_v2   -k flp --outputs output_bbox/BiasAdd,output_cov/Sigmoid --data_type fp16 -o rn10_70k_pruned_fp16.etlt data/resnet10/pruned/resnet10_nopool_bn_detectnet_v2_pruned.tlt
```

Once ready, the exported model can be connected to a Deepstream app. [Here](https://drive.google.com/open?id=14O8E5okYmYBp2fqL8wn75Y83OKHMAc21) is the link to config files for running the sample Deepstream app with the exported model. This sample app is to be launched as `deepstream-app -c config_deepstream.txt`

### Results
I was able to achieve 0.71 mAP on validation setm running at  28 FPS (35 ms). Qualitative results can be seen from videos and images below, or evaluated by running the inference command mentioned above on the test images attached to the root of the repository.

[download video here](https://drive.google.com/open?id=1PYMd4BkBKSSPxMtMHGAvi6uHiJrWtPOl)

[test image results](https://drive.google.com/open?id=1KtczIrgb7nTKAkedNODnuG_l7r6FEEy0)


### Weight links
1. pruned resnet10 model exported to fp16 for use in Deepstream app [link](https://drive.google.com/open?id=1bmC-SXc7O4h53h1rWjH80cvJUdtcEdVh)
2. Original resnet10 model in .tlt format to be used inside TLT [link](https://drive.google.com/open?id=1ISF-OsppMdK7MLMR8XiQ2qs_rIZ8QCq3)

