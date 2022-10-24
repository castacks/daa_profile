# DAA Profiling
Profiling Code for Multi-Camera DAA Inference

## Instructions to run the Profiling Code

Please download the images and model from Google Drive: [Google Drive Link](https://drive.google.com/drive/folders/1dViPz_SFTUZUspZlKZDvFgA2jTTvNIyL?usp=sharing)

Unzip the files to any location and change the `input_dir` and `models_dir` parameters of `profile.yaml` accordingly.

Please use `profile.yaml` to change TensorRT and Torch2TRT parameters.

The profiling code can be run using the following command:

```
python profile.py profile.yaml
```
