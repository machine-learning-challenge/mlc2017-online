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
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  flags.DEFINE_bool(
      "rendering", False,
      "If True, this will render the environment but drastically slow down training")

  flags.DEFINE_integer("export_model_steps", 10,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")

  flags.DEFINE_integer("total_episodes", 1000,
                       "Number of episodes")

  flags.DEFINE_integer("batch_size", 100,
                       "Size of batch for models that do parameter "
                       "updates in batches")

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def remove_training_directory(train_dir, task):
  """Removes the training directory."""
  try:
    logging.warning(
        "%s: Removing existing train directory.",
        task_as_string(task))
    gfile.DeleteRecursively(train_dir)
  except:
    logging.error(
        "%s: Failed to delete directory " + train_dir +
        " when starting a new model. Please delete it manually and" +
        " try again.", task_as_string(task))

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

  if is_master and FLAGS.start_new_model:
    remove_training_directory(train_dir, task)


  if not cluster or task.type == "master" or task.type == "worker":
    env = gym.make('CartPole-v0')
    model = find_class_by_name(FLAGS.model,
                               [models])()

    batch_size = FLAGS.batch_size # every how many episodes to do a param update?
    last_model_export_step = 0
    export_model_steps = FLAGS.export_model_steps

    with tf.Graph().as_default() as graph:
      meta_filename = get_meta_filename(FLAGS.start_new_model, train_dir, task)
      if meta_filename:
        logging.warning("using saved model %s", meta_filename)
        saver = recover_model(task, meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          global_step = tf.Variable(0, trainable=False, name="global_step")
          local_device_protos = device_lib.list_local_devices()
          gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
          num_gpus = len(gpus)
          if num_gpus > 0:
            logging.warning("Using the following GPUs to train: " + str(gpus))
            num_towers = num_gpus
            device_string = '/gpu:%d'
          else:
            logging.warning("No GPUs found. Training on CPU.")
            num_towers = 1
            device_string = '/cpu:%d'

          for i in range(num_towers):
            with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
              with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus!=1 else "/gpu:0")):
                results = model.build_graph(global_step)
                model.add_to_collection(results)

          model.collect()
          tf.add_to_collection("global_step", global_step)
          saver = tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)

        init = tf.global_variables_initializer()
        global_step = tf.get_collection("global_step")[0]
        model.get_collection(global_step)

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
    running_reward = None
    reward_sum = 0
    episode_number = 1
    total_episodes = FLAGS.total_episodes
    D = 4 #input dimensionality

    logging.warning("%s: Starting managed session.", task_as_string(task))
    with sv.managed_session(target, config=config) as sess:
      rendering = FLAGS.rendering
      observation = env.reset() # Obtain an initial observation of the environment

      model.before(sess)

      while episode_number <= total_episodes:
        if rendering:
          env.render()
          time.sleep(1./24)

        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        action = model.get_action(sess, x)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        model.after_action(sess, reward, info)
        if done:
          episode_number += 1

          global_step_val = model.after_episode(sess)

          # If we have completed enough episodes, then update the policy network with our gradients.
          if episode_number % batch_size == 0:
            model.after_batch(sess)

            # Give a summary of how well our network is doing for each batch of episodes.
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            logging.info('Global step %d. Average reward for episode %f.  Total average reward %f.' % (global_step_val, reward_sum/batch_size, running_reward/batch_size))
            if reward_sum/batch_size > 200:
              logging.info("Task solved in",episode_number,'episodes!')
              break

            reward_sum = 0

          time_to_export = ((last_model_export_step == 0) or
                            (global_step_val - last_model_export_step
                             >= export_model_steps))

          if is_master and time_to_export:
            last_checkpoint = saver.save(sess, sv.save_path, global_step_val)
            last_model_export_step = global_step_val

          observation = env.reset()

      if is_master:
        last_checkpoint = saver.save(sess, sv.save_path, global_step_val)
        last_model_export_step = global_step_val

      model.after()

if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  app.run()
