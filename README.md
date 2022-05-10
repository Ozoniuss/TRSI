# Implementing a Machine Learning classificatino model using Transfer Learning

This repository implements a machine learning model that predicts the classification of a flower image in 5 different categories. The model is implemented using the [MobileNet v2 Feature Vector](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) to which an output layer with 5 neurons is attached.

To run this code, Python 3 is required. Other requirements are listed in the `requirements.txt` file. To install them, simply run:

```bash
pip install -r requirements.txt
```

It is advised to set up a separate virtual environment for this beforehand to ensure that there are no dependency conflicts with other packages in the global environment:

```bash
python3 -m venv <env_name>
```

Run the `activate` script found in `<env_name>/Scripts` to activate and use the virtual environment. In bash terminals, Python will display the environment name when activated.

Alternatively, follow the tutorial [here](https://docs.python.org/3/library/venv.html) for installation. This project uses `virtualenv` but it is also possible to use `conda` as a package manager and virtual environment.

In order to run TensorFlow, the [NVIDIA CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive) must be installed, as well as the [NVIDIA cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse810-111) library. The .dll algorithm binaries fron the cuDNN library must pe copied to the CUDA Toolkit binaries for training to work.