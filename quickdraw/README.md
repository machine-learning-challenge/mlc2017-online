For English version, please check [English](README.en.md).

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

본 태스크에서, 여러분은 [Quick, Draw](https://quickdraw.withgoogle.com/) doodle을 인식하는 알고리즘을 개발해야 합니다.

Quick, Draw는 구글이 개발한 온라인 게임으로 유저가 무엇을 그리고 있는지 뉴럴 네트워크를 통하여 추측합니다. 본 태스크에서 여러분은 비슷한 역할을 하는 모델을 고안하셔야 하되, 문제의 범위는 10 카테고리로 제한됩니다. 데이터는 png형태로 제공됩니다.

## Input Data Format
인풋 데이터는 Google Cloud상에 있습니다. 다음 명령어를 통하여 데이터를 모두 다운로드 하실 수 있습니다:
```
gsutil cp -r gs://kmlc_test_train_bucket/quickdraw ./
```

훈련의 경우 Google Cloud상에서 진행하는 것을 권장합니다. 하지만 검증 및 테스트 데이터의 사이즈는 충분히 작기 때문에 로컬에서도 진행할 수 있으리라 생각합니다.

* Training data: gs://kmlc_test_train_bucket/quickdraw/*.tfrecords
* Validation data: gs://kmlc_test_train_bucket/quickdraw/validation/valid.tfrecords
* Test data: gs://kmlc_test_train_bucket/quickdraw/test/test.tfrecords

훈련 데이터와 교차검증 데이터는 같은 포맷으로 작성되어 있으며, 각각의 레코드는 두 개의 피처로 이뤄져 있습니다:
* image: byte list의 형태이며 내용은 png 포멧으로 저장된 이미지
* label: in64의 형태이며 내용은 해당 이미지의 레이블

Test data의 포멧은 조금 다릅니다:
* image: byte list의 형태이며 내용은 png 포멧으로 저장된 이미지
* image_id: byte list의 형태이며 내용은 해당 테스트 데이터의 id

*readers.py* 파일에서 예제 코드는 *tf.image.decode_png(image_str_tensor, channels=1)* 을 호출하여 이미지를 디코드하며 이미지는 흰색과 검은색밖에 없기 때문에 channels=1 인자를 사용합니다.

각 카테고리는 0부터 9까지 중 하나의 레이블이 매겨져 있으며, 각 레이블의 의미는 다음과 같습니다:
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

각 카테고리에 대하여 최소 75000개의 훈련 데이터가 있으며 2500개의 교차검증 데이터 및 테스트 데이터가 각각 있습니다.

## What you need to do
* losses.py 파일 안에서 BaseLoss를 상속받는 클래스를 만들고, calculate_loss를 구현하세요.
* models.py 안에서 BaseModel을 상속받아 나만의 모델을 구현하세요.
* 필요하다면 readers.py 파일에서 데이터 전처리/리사이징 등을 구현하세요. *구현시, QuickdrawFeatureReader 및 QuickDrawTestFeatureReader에도 같은 전처리가 반드시 포함되어야 합니다.*
* train.py 와 readers.py 파일 안에서 배치 사이즈나 학습 레이트와 같은 인자들을 수정하세요.
* 이상이 기본적인 가이드라인이나, 필요하다면 소스 코드의 어떤 부분도 자유롭게 수정하셔도 상관없습니다.

## Running Code in Google Cloud
훈련 데이터의 크기 떄문에라도, Google Cloud상에서 모델 훈련을 진행하는 것을 추천해드립니다. --model 인자와 기타 경로 등은 작업하시는 경로에 맞추어 수정해주세요.

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

여러 개의 Job들을 동시에 훈련시키는 것도 가능하며, tensorboard를 이용하면 Job들의 퍼포먼스를 시각화하여 비교하실 수 있습니다.

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

다음 gcloud 명령들은 모두 *저장소 루트* 경로에서 실행되어야 합니다.
현재 디렉토리의 위치는 명령어 'ls'를 통하여 확인하실 수 있습니다.

모델 개발 도중, 간단한 이슈들을 확인하기 위해 클라우드에 올릴 필요 없이 바로 로컬에서 테스트 해 보실 수 있습니다.
`gcloud beta ml local` 와 같은 명령어를 통해 이러한 일들을 할 수 있습니다.
다음 예제 커맨드에서 --model을 작업하신 모델 이름으로, data pattern과 train dir을 작업 환경에 맞도록 수정해주시기 바랍니다.

### Training Locally
```
gcloud ml-engine local train --package-path=quickdraw --module-name=quickdraw.train \
-- --train_data_pattern='TRAINING_DATA_LOCATION/*' --train_dir=/tmp/quickdraw_train \
--model=LogisticModel --start_new_model
```

### Evaluation Using Validation
교차검증 데이터를 이용해서 여러분의 모델을 평가 및 테스트하실 수 있습니다.
```
gcloud ml-engine local train --package-path=quickdraw --module-name=quickdraw.eval \
 -- --eval_data_pattern='VALIDATION_DATA_LOCATION/*' --train_dir=/tmp/quickdraw_train \
 --model=LogisticModel --run_once
```

### Generate Predictions on Test Data
inference.py 를 호출하여 label을 예측하신 뒤, 결과 파일을 Kaggle에 제출해주세요.
```
gcloud ml-engine local train --package-path=quickdraw --module-name=quickdraw.inference \
 -- --input_data_pattern='TEST_DATA_LOCATION/*' --train_dir=/tmp/quickdraw_train \
 --output_file=labels.csv
```
