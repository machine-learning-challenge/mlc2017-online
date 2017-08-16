# Cartpole Tensorflow Starter Code

한국어 버전은 [Korean](README.md)를 참고해주세요.

This repo contains starter code for training and evaluating machine learning
models over the cartpole dataset.

The code gives an end-to-end working example for reading the dataset, training a
TensorFlow model, and evaluating the performance of the model. Out of the box,
you can train several [model architectures](#overview-of-models) over features. 
The code can easily be extended to train your own custom-defined models.

It is possible to train and evaluate on cartpole in two ways: on Google Cloud
or on your own machine. This README provides instructions for both.

## Table of Contents
* [Overview](#overview)
   * [Description](#description)
   * [What you need to do](#what-you-need-to-do)
* [Running on Google's Cloud Machine Learning Platform](#running-on-googles-cloud-machine-learning-platform)
   * [Requirements](#requirements)
   * [Testing Locally](#testing-locally)
   * [Training on the Cloud](#training-on-cloud)
   * [Evaluation and Inference](#evaluation-and-inference)
   * [Accessing Files on Google Cloud](#accessing-files-on-google-cloud)
   * [Using Larger Machine Types](#using-larger-machine-types)
* [Running on Your Own Machine](#running-on-your-own-machine)
   * [Requirements](#requirements-1)
* [Overview of Models](#overview-of-models)
* [Overview of Files](#overview-of-files)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Submission](#submission)
   * [Misc](#misc)
* [About This Project](#about-this-project)

## Overview

### Description

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

![screen shot](https://cdn-images-1.medium.com/max/1600/1*oMSg2_mKguAGKy1C64UFlw.gif)

This simple problem has only 2 Actions:

|Num|Action|
|---|------|
|0|Push cart to the left|
|1|Push cart to the right|

It has only 4 observations:

|Num|Observation|Min|Max|
|---|-----------|---|---|
|0|Cart Position|-2.4|2.4|
|1|Cart Velocity|-Inf|Inf|
|2|Pole Angle|-41.8°|41.8°|
|3|Pole Velocity At Tip|-Inf|Inf|

At each iteration of an episode, a random position for the cart will be generated. Your algorithm
will provide either a 0 or 1 depending on which direction the cart should be moved. For each
iteration that the pole is upright, your algorithm will receive a reward of 1. You will 
then receive an observation which consists of a 4-tuple containing position, velocity, angle
and pole velocity. Use these observations to train and take a new action (0=left or 1=right).
When the episode is complete (the pole has fallen past the point of recovery or 200 steps have been reached)
the environment will remain until reset.

You can find more information about this problem [here](https://gym.openai.com/docs)

### What you need to do

First following the details in running to determine how to perform training, evaluation and inference.

Then you should really only need to modify [models.py](models.py). There are 2 models which have been
provided as a tutorial.

1. build_graph is a function that generates a tensorflow model.
2. add_to_collection is where you will save all variables necessary to instantiate a saved model.
3. get_collection will restore a saved model. 
4. before is run before any training, instantiate any global training variables here.
5. get_action returns an action.
6. after_action is run immediately following the action, so this is where you get to see the next observation.
7. after_episode is run after the episode is complete.
8. after_batch is run after a batch of N episodes, where N is up to you (set in the flag --batch_size)
9. after is run after training is compete
10. collect is run after the model is built, you may add global variables that are not part of your model here, and retrieve them in get_collection.

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

#### Enviroment

You must also have [OpenAIGym](https://gym.openai.com/) python enviroment cart-pole installed.

To install openai gym

```sh
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
```

### Testing Locally
All gcloud commands should be done from the directory *immediately above* the
source code. You should be able to see the source code directory if you
run 'ls'.

As you are developing your own models, you will want to test them
quickly to flush out simple problems without having to submit them to the cloud.
You can use the `gcloud beta ml local` set of commands for that.

Most commands will run in less than an hour running locally, but your time may
vary.

Here is an example command line:

```sh
gcloud ml-engine local train \
--package-path=cartpole --module-name=cartpole.train -- \
--train_dir=/tmp/kmlc_cartpole_train --model=PolicyGradient --start_new_model
```

To view the environment, you can also choose to turn rendering on
provided you have a display available

```sh
gcloud ml-engine local train \
--package-path=cartpole --module-name=cartpole.train -- \
--train_dir=/tmp/kmlc_cartpole_train --model=PolicyGradient --start_new_model \
--rendering
```

But note that this requires the installed rendering libraries.
If you installed openaigym with pip install -e ., you will need to run

```sh
pip install -e '.[classic_control]'
```

from the top-level directory of the location of where you downloaded gym.



Once your model is working locally, you can scale up on the Cloud
which is described below.

### Training on the Cloud

The following commands will train a model on Google Cloud

```sh
BUCKET_NAME=gs://${USER}_kmlc_cartpole_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=kmlc_cartpole_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=cartpole --module-name=cartpole.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=cartpole/cloudml-gpu.yaml \
-- \
--model=PolicyGradient \
--train_dir=$BUCKET_NAME/kmlc_cartpole_train
```

In the 'gsutil' command above, the 'package-path' flag refers to the directory
containing the 'train.py' script and more generally the python package which
should be deployed to the cloud worker. The module-name refers to the specific
python script which should be executed (in this case the train module).

It may take several minutes before the job starts running on Google Cloud.
When it starts you will see outputs like the following:

```
Global step 100. Average reward for episode 142.  Total average reward 184.34.
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
JOB_TO_EVAL=kmlc_cartpole_train_pgradient_model
JOB_NAME=kmlc_cartpole_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=cartpole --module-name=cartpole.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=cartpole/cloudml-gpu.yaml \
-- \
--model=PolicyGradient \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL}
```

Note the confusing use of 'training' in the above gcloud commands. Despite the
name, the 'training' argument really just offers a cloud hosted
python/tensorflow service. From the point of view of the Cloud Platform, there
is no distinction between our training and inference jobs. The Cloud ML platform
also offers specialized functionality for prediction with
Tensorflow models, but discussing that is beyond the scope of this readme.


### Submission

Here's how to submit your model to kaggle.

1. Edit conf.py to add your username and password. Then
2. Make sure you have accepted the rules at https://inclass.kaggle.com/c/kmlc-challenge-2-cartpole/rules
3. Choose compete as yourself or as a team
4. Submit the submission job

```sh
JOB_TO_EVAL=kmlc_cartpole_train_pgradient_model
JOB_NAME=kmlc_cartpole_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=cartpole --module-name=cartpole.submit \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=cartpole/cloudml-gpu.yaml \
-- \
--model=PolicyGradient \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL}
```

Check on the kaggle website for your results to be posted.

If you want to use a different model or a standalone version, this is fine, you simply
need to replace your use of the gym env with env_wrapper.py in this code. This will make
a connection to a remote server with the environment running. The submission 
sends your results to kaggle through the remote server.

Again, first modify conf.py with username and password.
Then modify your code and replace any usage of env with your .

``` python
    import conf
    import env_wrapper
    
    # some code
    
    # replace env = gym.make('CartPole-v0') with 
    env = env_wrapper.Service()

    # for 200 episodes, run your code
    
    # finally
    env.submit(conf.kaggle_user, conf.kaggle_passwd)
```

### Accessing Files on Google Cloud

You can browse the storage buckets you created on Google Cloud, for example, to
access the trained models, prediction CSV files, etc. by visiting the
[Google Cloud storage browser](https://console.cloud.google.com/storage/browser).


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

It also

Training
```sh
python train.py --train_dir=/tmp/cartpole_train --model=PolicyGradient --start_new_model
```

Validation

```sh
python eval.py --train_dir=/tmp/cartpole_train --model=PolicyGradient
```

Generating submission

First, edit conf.py to add your username and password. Then

```sh
python submit.py --train_dir=/tmp/cartpole_train
```

## Overview of Models

This sample code contains implementation of a policy gradient model

*   `PolicyGradient`: Use a final batch update with discounted rewards from the final state.
                     
                     
## Overview of Files

### Training
*   `train.py`: The primary script for training models.
*   `models.py`: Contains the sample model.

### Evaluation
*   `eval.py`: The primary script for evaluating models.

### Submission
*   `submit.py`: Submits the result to kaggle

### Misc
*   `README.md`: This documentation.

## About This Project
This project is meant help people quickly get started working with the 
[cartpole](LINK_TO_KMLC_SITE) dataset.
This is not an official Google product.
