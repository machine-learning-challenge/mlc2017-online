# Cats vs Dogs Tensorflow Starter Code

한국어 버전은 [Korean](README.md)을 참고해주세요.

This repo contains starter code for training and evaluating machine learning
models over the cats_vs_dogs dataset.

The code gives an end-to-end working example for reading the dataset, training a
TensorFlow model, and evaluating the performance of the model. Out of the box,
you can train several [model architectures](#overview-of-models) over features. 
The code can easily be extended to train your own custom-defined models.

It is possible to train and evaluate on Cats vs Dogs in two ways: on Google Cloud
or on your own machine. This README provides instructions for both.

## Table of Contents
* [Overview](#overview)
   * [Description](#description)
   * [What you need to do](#what-you-need-to-do)
* [Running on Google's Cloud Machine Learning Platform](#running-on-googles-cloud-machine-learning-platform)
   * [Requirements](#requirements)
   * [Testing Locally](#testing-locally)
   * [Training on the Cloud](#training-on-features)
   * [Evaluation and Inference](#evaluation-and-inference)
   * [Inference Using Batch Prediction](#inference-using-batch-prediction)
   * [Accessing Files on Google Cloud](#accessing-files-on-google-cloud)
   * [Using Larger Machine Types](#using-larger-machine-types)
* [Running on Your Own Machine](#running-on-your-own-machine)
   * [Requirements](#requirements-1)
* [Overview of Models](#overview-of-models)
* [Overview of Files](#overview-of-files)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Inference](#inference)
   * [Misc](#misc)
* [About This Project](#about-this-project)

## Overview

### Description

You are tasked with creating an algorithm that learns to classify images of cats or dogs.

<img src="https://cdn.pixabay.com/photo/2017/04/03/17/35/animals-2198994_960_720.jpg" height="350"/>

### What you need to do

First follow the details in running to determine how to perform training, evaluation and inference.

Then you should really only need to modify [cvd_models.py](cvd_models.py) and [losses.py](losses.py). There are 2 models which have been
provided as a tutorial.

1. You should create your own class which extends models.BaseModel and implements create_model
2. create_model generates your own network and returns a dict with the key predictions e.g. ``{"predictions":predictions}``
3. You should create your own class which extends BaseLoss and implements calculate_loss. 
4. calculate loss takes prediction and labels and returns a tensorflow variable which contains the differentiable loss.
5. During training you should specify --model=YourModelClass and --label_loss=YourLossFunctionClass
6. You should make certain that the output layer of your network matches the prediction input of your loss function!

You should not need to modify any other aspect, but you may find that there are other functions you desire. You may
add functions to your model and modify train.py for your needs. 

Good luck!

## Running on Google's Cloud Machine Learning Platform

### Requirements

This option requires you to have an appropriately configured Google Cloud
Platform account. To create and configure your account, please make sure you
follow the instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

Please also verify that you have Python 2.7+ and Tensorflow 1.0.1 or higher
installed by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

### Testing Locally
All gcloud commands should be done from the directory *immediately above* the
source code. You should be able to see the source code directory if you
run 'ls'.

As you are developing your own models, you will want to test them
quickly to flush out simple problems without having to submit them to the cloud.
You can use the `gcloud beta ml local` set of commands for that.
Here is an example command line:

```sh
gcloud ml-engine local train \
--package-path=cats_vs_dogs --module-name=cats_vs_dogs.train -- \
--train_data_pattern='gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/train/*' \
--train_dir=/tmp/kmlc_cvd_train --model=LogisticModel --start_new_model
```

You might want to download some training shards locally to speed things up and
allow you to work offline. The command below will copy 10 out of the 4096
training data files to the current directory.

```sh
# Downloads 344MB of data.
gsutil cp gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/train/train-0000[0-9]*-of-00014 ./
```
Once you download the files, you can point the job to them using the
'train_data_pattern' argument (i.e. instead of pointing to the "gs://..."
files, you point to the local files).

Once your model is working locally, you can scale up on the Cloud
which is described below.

### Training on the Cloud

The following commands will train a model on Google Cloud

```sh
BUCKET_NAME=gs://${USER}_kmlc_cvd_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=kmlc_cvd_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=cats_vs_dogs --module-name=cats_vs_dogs.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=cats_vs_dogs/cloudml-gpu.yaml \
-- --train_data_pattern='gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/train/*' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/kmlc_cvd_train_logistic_model
```

In the 'gsutil' command above, the 'package-path' flag refers to the directory
containing the 'train.py' script and more generally the python package which
should be deployed to the cloud worker. The module-name refers to the specific
python script which should be executed (in this case the train module).

It may take several minutes before the job starts running on Google Cloud.
When it starts you will see outputs like the following:

```
training step 270| Hit@1: 0.68 PERR: 0.52 Loss: 638.453
training step 271| Hit@1: 0.66 PERR: 0.49 Loss: 635.537
training step 272| Hit@1: 0.70 PERR: 0.52 Loss: 637.564
```

At this point you can disconnect your console by pressing "ctrl-c". The
model will continue to train indefinitely in the Cloud. Later, you can check
on its progress or halt the job by visiting the
[Google Cloud ML Jobs console](https://console.cloud.google.com/ml/jobs).

You can train many jobs at once and use tensorboard to compare their performance
visually.

```sh
tensorboard --logdir=$BUCKET_NAME --port=8080
```

Once tensorboard is running, you can access it at the following url:
[http://localhost:8080](http://localhost:8080).
If you are using Google Cloud Shell, you can instead click the Web Preview button
on the upper left corner of the Cloud Shell window and select "Preview on port 8080".
This will bring up a new browser tab with the Tensorboard view.

### Evaluation and Inference
Here's how to evaluate a model on the validation dataset:

```sh
JOB_TO_EVAL=kmlc_cvd_train_logistic_model
JOB_NAME=kmlc_cvd_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=cats_vs_dogs --module-name=cats_vs_dogs.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=cats_vs_dogs/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/validation/*' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=True
```

And here's how to perform inference with a model on the test set:

```sh
JOB_TO_EVAL=kmlc_cvd_train_logistic_model
JOB_NAME=kmlc_cvd_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=cats_vs_dogs --module-name=cats_vs_dogs.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=cats_vs_dogs/cloudml-gpu.yaml \
-- --input_data_pattern='gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/test/*' \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv
```

Note the confusing use of 'training' in the above gcloud commands. Despite the
name, the 'training' argument really just offers a cloud hosted
python/tensorflow service. From the point of view of the Cloud Platform, there
is no distinction between our training and inference jobs. The Cloud ML platform
also offers specialized functionality for prediction with
Tensorflow models, but discussing that is beyond the scope of this readme.

Once these job starts executing you will see outputs similar to the
following for the evaluation code:

```
examples_processed: 1024 | global_step 447044 | Batch Hit@1: 0.782 | Batch PERR: 0.637 | Batch Loss: 7.821 | Examples_per_sec: 834.658
```

and the following for the inference code:

```
num examples processed: 8192 elapsed seconds: 14.85
```

### Inference Using Batch Prediction
To perform inference faster, you can also use the Cloud ML batch prediction
service.

First, find the directory where the training job exported the model:

```
gsutil list ${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export
```

You should see an output similar to this one:

```
${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export/
${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export/step_1/
${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export/step_1001/
${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export/step_2001/
${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export/step_3001/
```

Select the latest version of the model that was saved. For instance, in our
case, we select the version of the model that was saved at step 3001:

```
EXPORTED_MODEL_DIR=${BUCKET_NAME}/kmlc_cvd_train_logistic_model/export/step_3001/
```

Start the batch prediction job using the following command:

```
JOB_NAME=kmlc_cvd_batch_predict_$(date +%Y%m%d_%H%M%S); \
gcloud ml-engine jobs submit prediction ${JOB_NAME} --verbosity=debug \
--model-dir=${EXPORTED_MODEL_DIR} --data-format=TF_RECORD \
--input-paths='gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/test/*' \
--output-path=${BUCKET_NAME}/batch_predict/${JOB_NAME} --region=us-east1 \
--runtime-version=1.2 --max-worker-count=10
```

You can check the progress of the job on the
[Google Cloud ML Jobs console](https://console.cloud.google.com/ml/jobs). To
have the job complete faster, you can increase 'max-worker-count' to a
higher value.

Once the batch prediction job has completed, turn its output into a submission
in the CVS format by running the following commands:

```
# Copy the output of the batch prediction job to a local directory
mkdir -p /tmp/batch_predict/${JOB_NAME}
gsutil -m cp -r ${BUCKET_NAME}/batch_predict/${JOB_NAME}/* /tmp/batch_predict/${JOB_NAME}/

# Convert the output of the batch prediction job into a CVS file ready for submission
python cats_vs_dogs/convert_prediction_from_json_to_csv.py \
--json_prediction_files_pattern="/tmp/batch_predict/${JOB_NAME}/prediction.results-*" \
--csv_output_file="/tmp/batch_predict/${JOB_NAME}/output.csv"
```

### Accessing Files on Google Cloud

You can browse the storage buckets you created on Google Cloud, for example, to
access the trained models, prediction CSV files, etc. by visiting the
[Google Cloud storage browser](https://console.cloud.google.com/storage/browser).

Alternatively, you can use the 'gsutil' command to download the files directly.
For example, to download the output of the inference code from the previous
section to your local machine, run:


```
gsutil cp $BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv .
```


### Using Larger Machine Types

Some complex models can take as long as a week to converge when
using only one GPU. You can train these models more quickly by using more
powerful machine types which have additional GPUs. To use a configuration with
4 GPUs, replace the argument to `--config` with `cats_vs_dogs/cloudml-4gpu.yaml`.
Be careful with this argument as it will also increase the rate you are charged
by a factor of 4 as well.

## Running on Your Own Machine

### Requirements

The starter code requires Tensorflow. If you haven't installed it yet, follow
the instructions on [tensorflow.org](https://www.tensorflow.org/install/).
This code has been tested with Tensorflow 1.0.1. Going forward, we will continue
to target the latest released version of Tensorflow.

Please verify that you have Python 2.7+ and Tensorflow 1.0.1 or higher
installed by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

Downloading files
``` sh
gsutil cp gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/train/train-* .
gsutil cp gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/validation/validation-* .
gsutil cp gs://kmlc_test_train_bucket/cats_vs_dogs/tfrecords/test/test-* .
```


Training
```sh
python train.py --train_data_pattern='/path/to/training/files/*' --train_dir=/tmp/cvd_train --model=LogisticModel --start_new_model
```

Validation

```sh
python eval.py --eval_data_pattern='/path/to/validation/files' --train_dir=/tmp/cvd_train --model=LogisticModel --run_once=True
```

Generating submission

```sh
python inference.py --output_file=/path/to/predictions.csv --input_data_pattern='/path/to/test/files/*' --train_dir=/tmp/cvd_train
```

## Overview of Models

This sample code contains implementation of the logistic model:

*   `LogisticModel`: Linear projection of the output features into the label
                     space, followed by a sigmoid function to convert logit
                     values to probabilities.
                     
## Overview of Files

### Training
*   `train.py`: The primary script for training models.
*   `losses.py`: Contains definitions for loss functions.
*   `models.py`: Contains the base class for defining a model.
*   `cvd_models.py`: Contains definitions for models that take
                             aggregated features as input.
*   `export_model.py`: Provides a class to export a model during training
                       for later use in batch prediction.
*   `readers.py`: Contains definitions for the dataset and Frame
                  dataset readers.

### Evaluation
*   `eval.py`: The primary script for evaluating models.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating
                                       average precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean
                                            average precision.

### Inference
*   `inference.py`: Generates an output file containing predictions of
                    the model over a set of images.

### Misc
*   `README.md`: This documentation.
*   `utils.py`: Common functions.
*   `convert_prediction_from_json_to_csv.py`: Converts the JSON output of
        batch prediction into a CSV file for submission.

## About This Project
This project is meant help people quickly get started working with the 
[cats_vs_dogs](LINK_TO_KMLC_SITE) dataset.
This is not an official Google product.
