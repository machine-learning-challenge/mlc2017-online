한국어 버전은 [Korean](README.md)을 참고해주세요.

Table of Contents
=================

   * [Table of Contents](#table-of-contents)
   * [Quick, Draw](#quick-draw)
      * [Input Data Format](#input-data-format)
      * [What you need to do](#what-you-need-to-do)
      * [Running Code in Google Cloud](#running-code-in-google-cloud)
         * [Training in Cloud](#training-in-cloud)
         * [Evaluating the Model in Cloud](#evaluating-the-model-in-cloud)
         * [Generate Predictions in Cloud](#generate-predictions-in-cloud)
      * [Running Code Locally](#running-code-locally)
         * [Training Locally](#training-locally)
         * [Evaluation Using Validation](#evaluation-using-validation)
         * [Generate Predictions on Test Data](#generate-predictions-on-test-data)

# Quick, Draw

In this problem, you will attempt to recognize the 
[Quick, Draw](https://quickdraw.withgoogle.com/) doodles.

Quick, Draw is an online game developed by Google where a neural network is used 
to guess what a user is drawing. In this problem, you will be asked to do something similar,
except there are only 10 categories involved. The data are rasterized to png format. 

## Input Data Format
All the data are available on Google Cloud. You can download all data using
```
gsutil cp -r gs://kmlc_test_train_bucket/quickdraw ./
```

Most likely you want to train your model in Google Cloud, but validation and test data are 
small enough that you may also want to run them locally.

* Training data: gs://kmlc_test_train_bucket/quickdraw/training/*.tfrecords
* Validation data: gs://kmlc_test_train_bucket/quickdraw/validation/valid.tfrecords
* Test data: gs://kmlc_test_train_bucket/quickdraw/test/test.tfrecords

Training data and cross validation data share the same format, where each record consists of two features:
* image: a byte list feature that stores the png encoding of the image
* label: an int64 feature on the label of the image

Test data format is a little different:
* image: a byte list feature that stores the png encoding of the image
* image_id: a byte list feature represents the id of a test data

In *readers.py*, the sample code call *tf.image.decode_png(image_str_tensor, channels=1)* to decode the image,
and channels=1 is used since the images only contain black and white colors.

Each category is assigned to a label from 0 to 9, with
* 0 -> ants
* 1 -> boomerang
* 2 -> cake
* 3 -> lion
* 4 -> monkey
* 5 -> pig
* 6-> scissor
* 7 -> skull
* 8 -> television
* 9 -> traffic_light.

For each category, there are at least 75000 training data, and 2500 cross validation data and test data respectively.

## What you need to do
* You should create your own class which extends BaseLoss and implements calculate_loss in losses.py
* You should create your own model which extends BaseModel in models.py
* Add data preprocessing/resizing if needed in readers.py. *Make sure that the same preprocessing is applied in both QuickDrawFeatureReader and QuickDrawTestFeatureReader*
* Adjust parameters in train.py and readers.py such as batch size and learning rate
* In addition, you are free to modify any part of the source code in any file

## Running Code in Google Cloud
Consider the size of training data, most likely you want to train in Google Cloud. Replace --model with your model name and modify the directory paths accordingly.

### Training in Cloud
```
BUCKET_NAME=gs://${USER}_kmlc_quickdraw_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=kmlc_quickdraw_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=quickdraw --module-name=quickdraw.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=quickdraw/cloudml-gpu.yaml \
-- --train_data_pattern='gs://kmlc_test_train_bucket/quickdraw/training/*.tfrecords' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/kmlc_quickdraw_train_logistic_model
```

You can train many jobs at once and use tensorboard to compare their performance visually.
```
tensorboard --logdir=$BUCKET_NAME --port=8080
```

### Evaluating the Model in Cloud
```
JOB_TO_EVAL=kmlc_quickdraw_train_logistic_model
JOB_NAME=kmlc_quickdraw_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=quickdraw --module-name=quickdraw.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=quickdraw/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://kmlc_test_train_bucket/quickdraw/validation/valid.tfrecords' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=True
```

### Generate Predictions in Cloud
```
JOB_TO_EVAL=kmlc_quickdraw_train_logistic_model
JOB_NAME=kmlc_quickdraw_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=quickdraw --module-name=quickdraw.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=quickdraw/cloudml-gpu.yaml \
-- --input_data_pattern='gs://kmlc_test_train_bucket/quickdraw/test/test.tfrecords' \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv
```

## Running Code Locally
<span style="color:red">Skeleton code may still contain errors and is being updated. </span>

All gcloud commands should be done from the directory *immediately above* the
source code. You should be able to see the source code directory if you
run 'ls'.

As you are developing your own models, you will want to test them
quickly to flush out simple problems without having to submit them to the cloud.
You can use the `gcloud beta ml local` set of commands for that.
Here is an example command line, you should replace --model with your model name,  and 
modify data pattern and train dir accordingly.

### Training Locally
```
gcloud ml-engine local train --package-path=quickdraw --module-name=quickdraw.train \
-- --train_data_pattern='TRAINING_DATA_LOCATION/*' --train_dir=/tmp/quickdraw_train \
--model=LogisticModel --start_new_model
```

### Evaluation Using Validation
You can evaluate and test your model using the cross validation data.
```
gcloud ml-engine local train --package-path=quickdraw --module-name=quickdraw.eval \
 -- --eval_data_pattern='VALIDATION_DATA_LOCATION/*' --train_dir=/tmp/quickdraw_train \
 --model=LogisticModel --run_once
```

### Generate Predictions on Test Data
You can generate the predictions using inference.py and submit the output file with the labels to Kaggle.
```
gcloud ml-engine local train --package-path=quickdraw --module-name=quickdraw.inference \
 -- --input_data_pattern='TEST_DATA_LOCATION/*' --train_dir=/tmp/quickdraw_train \
 --output_file=labels.csv
```
