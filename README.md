# Torch Multiwell Segmentation

![](img/demo.png)

This project contains a pretrained ML Model to detect wells and cell colony in multiwell plates.

> Note: for a simpler version of this software, that is only based on opencv and can be run completely client side in the browser, see: https://github.com/antielektron/cell_colony_detector

## Installation

```bash
pip install git+https://github.com/antielektron/torch_cell_colony_detector.git
```

the model is served as a [panel](https://panel.holoviz.org/) app, that you can run with

```bash
multiwell_dashboard
```

if you used pip from a python installation on non default system paths and don't have the command `multiwell_dashboard` available, you can run the app with

```bash
python -m torch_multiwell_segmentation.serve
```

## Development installation and Training

```bash
git clone https://github.com/antielektron/torch_cell_colony_detector.git
cd torch_cell_colony_detector
pip install -e .
```

The training notebooks are in the folder [Training](Training/) and can be run with jupyter notebook or jupyter lab.

If you want to train your own dataset, perform the following steps:

* replace the images in [Training/cell_culture_plate_images](Training/cell_culture_plate_images) with your own images
* clear the folder content of  X, Y and Z
* run the notebook [DatasetGeneration.ipynb](Training/DatasetGeneration.ipynb)
    * this notebook will randomly present you detected circle areas from potential wells and asks you whether they are wells or not. the images will be labeled accordingly and saved for the training.
* run the notebook [UNET-Training.ipynb](Training/UNET-Training.ipynb)
    * this notebook will train the model and save it
    * after that, copy the model generated in the same folder to [torch_multiwell_segmentation/data](torch_multiwell_segmentation/data)
* to test the model, you can run the notebook [Inference.ipynb](Training/Inference.ipynb)