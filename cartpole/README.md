# Cartpole Tensorflow Starter Code

For English version, please check [English](README.en.md).

본 저장소는 Cartpole 문제를 해결하기 위한 트레이닝과 머신러닝 모델 평가를 위한 기본적인 코드를 담고 있습니다.

본 저장소에 있는 코드는 데이터를 읽고, TensorFlow 모델을 훈련하고, 모델의 성능을 평가하는 일련의 과정에 대한 예제 코드입니다.
여러분은 데이터를 이용해서 [model architectures](#overview-of-models)에 주어진 것과 같은 여러 모델들을 훈련할 수 있습니다.
또한, 본 코드는 주어진 모델 외에도 여러분의 커스텀 모델에도 쉽게 적용될 수 있도록 디자인 되어 있습니다.

Cartpole을 훈련시키고 평가하는 방법에는 두 가지가 있습니다: Google Cloud 위에서 작업하실 수도 있고, 여러분의 로컬 머신에서 작업하실 수도 있습니다.
본 README는 양쪽 방법 모두를 다루고자 합니다.

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

마찰력 없는 트랙 위에 카트가 놓여있고, 카트 위의 한 점에 막대가 붙어 있습니다. 막대는 시작할 때 수직으로 서 있고, 본 태스크의 목표는 카트의 속도를 조절하여 막대가 넘어지지 않도록 최대한 오래 버티는 것입니다.

![screen shot](https://cdn-images-1.medium.com/max/1600/1*oMSg2_mKguAGKy1C64UFlw.gif)

본 문제는 매우 단 두가지의 액션만을 지원합니다:

|Num|Action|
|---|------|
|0|카트를 왼쪽으로 민다|
|1|카트를 오른쪽으로 민다|

이후 네 종류의 관찰 결과를 볼 수 있습니다:

|Num|Observation|Min|Max|
|---|-----------|---|---|
|0|카트 위치|-2.4|2.4|
|1|카트 속도|-Inf|Inf|
|2|막대 각도|-41.8°|41.8°|
|3|막대 끝부분의 속도|-Inf|Inf|

각 에피소드의 매 턴마다 카트의 위치가 랜덤하게 생성될 것입니다. 여러분의 알고리즘은 카트가 움직여야 하는 방향을 0 혹은 1로 반환해야 합니다. 막대가 똑바로 설 때마다 1 만큼의 보상을 얻게 됩니다. 그 후 위치, 속도, 각도, 그리고 막대의 속도를 저장하고 있는 4-튜플 포멧의 관찰 결과를 전달받습니다. 이 결과를 이용하여 훈련을 진행하고, 또 다음 액션을 결정하세요(0 = 왼쪽, 1 = 오른쪽). 한 에피소드가 끝날 때(막대가 회복 가능한 영역 밖으로 벗어나거나 200 턴을 버텨내었을 때), environment는 리셋 전까지 남아 있을 것입니다.

본 문제에 대한 좀 더 자세한 정보는 [이 곳](https://gym.openai.com/docs)에서 확인하실 수 있습니다.

### What you need to do

먼저, 아래에서 설명할 플랫폼 별 가이드를 따라서 훈련, 평가 및 예측 하는 방법에 대해서 숙지하시기 바랍니다.

그 후, 여러분이 변경해야 하는 파일은 [models.py](models.py) 입니다. 주어진 파일 안에는 이미 튜토리얼로써 2개의 모델이 주어져 있습니다.

1. build_graph 는 Tensorflow 모델을 생성하는 함수입니다.
2. add_to_collection은 저장된 모델을 인스턴스화하기 위해 필요한 모든 변수들을 저장하는 곳입니다.
3. get_collection은 저장된 모델을 불러옵니다.
4. before은 어떠한 종류의 훈련이든 훈련 전에 전에 실행되며, 훈련에 사용될 전역 변수들을 인스턴싱하는 곳입니다.
5. get_action은 액션을 반환합니다.
6. after_action은 액션 이후에 바로 실행되며, 이 곳에서 다음 관찰 결과를 볼 수 있습니다.
7. after_episode는 각 에피소드가 종료되고 나서 실행됩니다.
8. ater_batch는 N개의 에피소드 배치 수행 후에 실행되며, N은 전달 인자로써 조절 가능합니다. (--batch_size 플래그를 이용)
9. after은 트레이닝이 종료된 후에 실행됩니다.
10. collect는 모델이 만들어지고 나서 호출됩니다. 이 함수는 모델의 일부는 아니지만 추가하고 싶은 전역 변수가 있을 때 이 곳에 변수를 추가하여 get_collection에서 불러올 수 있게 합니다.

기본적으로 위 파일 이외에는 수정하지 않으셔도 태스크 수행에 문제는 없습니다. 하지만 추가적으로 기능이 필요하시다면 model, 그리고 train.py 에 추가적인 기능을 구현하셔도 좋습니다.

행운을 빕니다!

## Running on Google's Cloud Machine Learning Platform

### Requirements

먼저 여러분의 계정이 Google Cloud Platform을 사용할 수 있도록 세팅되어야 합니다.
본 [링크](https://cloud.google.com/ml/docs/how-tos/getting-set-up)를 따라서 계정 생성 및 설정을 진행해주세요.

또한, 다음 명령들을 실행하여 Python 2.7 이상, 그리고 TensorFlow 1.0.1 이상 버전이 설치되어 있는지 확인하시기 바랍니다:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

#### Enviroment

[OpenAIGym](https://gym.openai.com/)의 Cart-pole Python Environment를 필요로 합니다.

Open AI Gym 설치:

```sh
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
```

### Testing Locally
다음 gcloud 명령들은 모두 *저장소 루트* 경로에서 실행되어야 합니다.
현재 디렉토리의 위치는 명령어 'ls'를 통하여 확인하실 수 있습니다.

모델 개발 도중, 간단한 이슈들을 확인하기 위해 클라우드에 올릴 필요 없이 바로 로컬에서 테스트 해 보실 수 있습니다.
`gcloud beta ml local` 와 같은 명령어를 통해 사용하시면 되며, 예제는 다음과 같습니다:

대부분의 명령은 로컬에서 실행시 한 시간 미만으로 작업이 끝나지만, 이는 작업에 내용에 따라 다를 수 있습니다.

예제 코드는 다음과 같습니다:

```sh
gcloud ml-engine local train \
--package-path=cartpole --module-name=cartpole.train -- \
--train_dir=/tmp/kmlc_cartpole_train --model=PolicyGradient --start_new_model
```

작업 환경을 직접 보기 위해서는 렌더링을 켤 수도 있습니다:

```sh
gcloud ml-engine local train \
--package-path=cartpole --module-name=cartpole.train -- \
--train_dir=/tmp/kmlc_cartpole_train --model=PolicyGradient --start_new_model \
--rendering
```

위 렌더링 옵션을 켜기 위해서는 렌더링 라이브러리가 사전에 설치되어 있어야 합니다.
만약 Open AI Gym을 `python install -e .` 명령으로 설치하였다면 Gym을 다운받은 탑 디렉토리에서 다음 역시 실행하여야 합니다:

```sh
pip install -e '.[classic_control]'
```

로컬에서 잘 동작하는 것을 확인하였다면 이를 클라우드로 올려 좀 더 스케일을 키울 수 있습니다.

### Training on the Cloud

다음 명령어는 해당 모델을 Google Cloud 위에서 훈련시킬 것입니다. 다음 명령어들은 Google Cloud Shell위에서 실행되어야 합니다.

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

위 `gsutil` 명령어에서 'package-path' 플래그는 'train.py' 스크립트를 포함하고 있는 경로를 의미하며, 동시에 Cloud worker에 업로드 될 패키지를 의미하기도 합니다. 'module-name'은 실행되어야 할 파이선 스크립트를 지정하는 플래그입니다.(본 예제에서는 train module을 사용하고 있습니다.)

Google Cloud에서 업로드 후 job들이 실행되기까지는 잠시 시간이 걸립니다.
실행이 되고 나면 다음과 같은 메세지들을 보실 수 있습니다:

```
Global step 100. Average reward for episode 142.  Total average reward 184.34.
```

작업 도중 "ctrl-c" 를 눌러 콘솔에서 접속을 끊을 수 있습니다. 모델은 클라우드 상에서 독립적으로 계속 훈련이 진행되며, 해당 job에 대한 진행 상황을 확인하거나 멈추는 등의 작업은 [Google Cloud ML Jobs console](https://console.cloud.google.com/ml/jobs) 를 통해서 하실 수 있습니다.

여러 job들을 동시에 훈련시킬 수도 있으며, tensorboard를 통해서 모델들의 퍼포먼스를 시각화하여 보실 수 있습니다.

```sh
tensorboard --logdir=$BUCKET_NAME --port=8080
```

Tensorboard가 실행되고 나면 다음 명령어를 통해서 Tensorboard를 보실 수 있습니다: [http://localhost:8080](http://localhost:8080)
만약 Google Cloud Shell에서 실행하셨다면 콘솔 위쪽에 있는 Web Preview 버튼을 누르고 "Preview on port 8080" 메뉴를 통해서 Tensorboard를 보실 수 있습니다.

### Evaluation and Inference
다음은 모델을 Validation dataset위에서 평가하는 방법입니다:

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

위 gcloud 명령어에서 'training'이라는 부분이 다소 착각의 여지가 있습니다. 이름과는 다르게, 'training'이란 명령어는 클라우드가 호스팅하는 Python/Tensorflow 서비스를 제공하는 일을 합니다. Cloud Platform의 관점에서 보면 training과 inference는 아무 차이가 없는 일이기 때문입니다. Cloud ML Platform은 Tensorflow를 통한 예측을 위한 특별한 기능들을 제공하기는 하나 본 문서에서는 다루지 않겠습니다.

### Submission

Kaggle에 본 태스크의 결과를 제출하는 방법입니다.

1. conf.py 파일을 편집하여 Kaggle username과 password를 입력하세요.
2. https://inclass.kaggle.com/c/kmlc-challenge-2-cartpole/rules 에서 Kaggle rule에 동의해주세요.
3. 개인 또는 팀으로 참가할지를 결정해주세요.
4. 다음 커맨드를 통해 제출해주세요:

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

이후 Kaggle 웹사이트에서 결과가 제출되었는지 확인하시면 됩니다.

원하시면 다른 모델을 사용하시거나 Standalone 버전을 사용하실 수도 있습니다. env_warpper.py 안에 있는 gym env 부분을 원하시는 형태로 대체하시면 됩니다. 이 부분은 environment가 구동되고 있는 원격 서버에 접속하는 역할을 합니다. 여러분이 제출하신 결과는 이 서버를 통해서 Kaggle에 제출됩니다.

이 쪽 방법 역시, 먼저 conf.py 파일을 편집하여 Kaggle username과 암호를 일벽해주세요.
그 후 코드를 수정하시고 env부분을 여러분의 코드로 대체하시면 됩니다.

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

[Google Cloud storage browser](https://console.cloud.google.com/storage/browser)을 통해서 이전에 생성한 storage bucket을 직접 조회할 수 있습니다. 해당 버킷에 저장된 Trained model, CSV 파일 등을 직접 조회할 수 있습니다.

## Running on Your Own Machine

### Requirements

시작하기 전 Tensorflow가 준비되어 있어야 합니다. 만약 아직 설치하지 않으셨다면 [tensorflow.org](https://www.tensorflow.org/install/)에 쓰인 설명을 따라주세요.

다음 명령어를 통해 Python 2.7 이상, 그리고 Tensorflow 1.0.1 이상이 설치되어 있는지 확인하시기 바랍니다:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

Training
```sh
python train.py --train_dir=/tmp/cartpole_train --model=PolicyGradient --start_new_model
```

Validation

```sh
python eval.py --train_dir=/tmp/cartpole_train --model=PolicyGradient
```

Generating submission

conf.py 파일을 편집해 Kaggle username과 암호를 입력해주시기 바랍니다. 그리고 다음을 실행해주세요:

```sh
python submit.py --train_dir=/tmp/cartpole_train
```

## Overview of Models

This sample code contains implementation of a policy gradient model
다음 예제 코드는 [Policy gradient model](https://en.wikipedia.org/wiki/Gradient)의 구현체를 담고 있습니다.

*   `PolicyGradient`: 마지막 상태로부터 Reward를 소폭 반영하여 배치 업데이트를 통해 개선해 나가는 방법입니다.
                     
## Overview of Files

### Training
*   `train.py`: 모델 훈련을 위한 코어 로직을 담가ㅗ 있는 스크립트입니다.
*   `models.py`: 예시 모델을 담고 있습니다.

### Evaluation
*   `eval.py`: 평가를 위한 코어 로직을 담고 있는 스크립트입니다.

### Submission
*   `submit.py`: Kaggle에 결과를 제출하기 위한 스크립트입니다.

### Misc
*   `README.md`: 이 문서입니다.

## About This Project
This project is meant help people quickly get started working with the 
[cartpole](LINK_TO_KMLC_SITE) dataset.
This is not an official Google product.
