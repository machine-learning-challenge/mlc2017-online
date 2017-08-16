from pyglet.gl import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import models
import gym
import numpy as np
import json
import os
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.client import device_lib
import sys
import time

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/kmlc_model/",
                      "The directory to save the model files in.")

  flags.DEFINE_string(
      "model", "PolicyGradient",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")

  flags.DEFINE_bool(
      "rendering", False,
      "If set, this will render the environment but drastically slow down training")

  flags.DEFINE_integer("total_episodes", 1000,
                       "Number of episodes")
def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def get_meta_filename(start_new_model, train_dir, task):
  if start_new_model:
    logging.warning("%s: Flag 'start_new_model' is set. Building a new model.",
                 task_as_string(task))
    return None

  latest_checkpoint = tf.train.latest_checkpoint(train_dir)
  if not latest_checkpoint:
    logging.warning("%s: No checkpoint file found. Building a new model.",
                 task_as_string(task))
    return None

  meta_filename = latest_checkpoint + ".meta"
  if not gfile.Exists(meta_filename):
    logging.warning("%s: No meta graph file found. Building a new model.",
                   task_as_string(task))
    return None
  else:
    return meta_filename

def recover_model(task, meta_filename):
    logging.warning("%s: Restoring from meta graph file %s",
                 task_as_string(task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def main(unused_argv):
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)
  is_master = (task.type == "master" and task.index == 0)
  train_dir = FLAGS.train_dir
  D = 4 # input dimensionality

  if cluster:
    logging.warning("%s: Starting trainer within cluster %s.",
                 task_as_string(task), cluster.as_dict())
    server = start_server(cluster, task)
    target = server.target
    device_fn = tf.train.replica_device_setter(
        ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (task.type, task.index),
        cluster=cluster)
  else:
    target = ""
    device_fn = ""

  config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)


  if not cluster or task.type == "master" or task.type == "worker":
    env = gym.make('CartPole-v0')
    model = find_class_by_name(FLAGS.model,
                               [models])()

    with tf.Graph().as_default() as graph:
      meta_filename = get_meta_filename(False, train_dir, task)
      if meta_filename:
        logging.warning("using saved model %s", meta_filename)
        saver = recover_model(task, meta_filename)
      else:
        raise("meta file not found")

      with tf.device(device_fn):
        init = tf.global_variables_initializer()
        global_step = tf.get_collection("global_step")[0]
        model.get_collection(global_step)

    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    sv = tf.train.Supervisor(
        graph,
        logdir=train_dir,
        init_op=init,
        is_chief=is_master,
        global_step=global_step,
        save_model_secs=3600,
        save_summaries_secs=120,
        saver=saver)


    # Launch the graph
    drs = []
    running_reward_sum = 0
    reward_sum = 0
    episode_number = 0
    total_episodes = FLAGS.total_episodes

    logging.warning("%s: Starting managed session.", task_as_string(task))
    with sv.managed_session(target, config=config) as sess:

      rendering = FLAGS.rendering
      observation = env.reset() # Obtain an initial observation of the environment

      model.before(sess)
      while episode_number < total_episodes:
        if rendering == True:
          env.render()
          time.sleep(1./24)
        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation, [1, D])
        action = model.get_action(sess, x)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        running_reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        if done:
          episode_number += 1
          if episode_number % 100 == 0:
            logging.info('Reward for episode %d = %f.  Total average reward %f.' % (episode_number, reward_sum, running_reward_sum/episode_number))
          reward_sum = 0
          observation = env.reset()
          done = False

      logging.info('Total reward: %d, Avg reward: %f' % (running_reward_sum, running_reward_sum / total_episodes))

      avg_reward_sum = running_reward_sum / total_episodes
      summary_writer.add_summary(MakeSummary("GlobalStep/Eval_TotalRewardSum", running_reward_sum), running_reward_sum);
      summary_writer.add_summary(MakeSummary("GlobalStep/Eval_AvgRewardSum", avg_reward_sum), avg_reward_sum);
      summary_writer.flush()
      # submit via tf serving



if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  app.run()
