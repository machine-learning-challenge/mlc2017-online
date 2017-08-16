import tensorflow as tf
import numpy as np

class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()


class QLearning(BaseModel):

  def create_model():
    return QLearning()

  def build_graph(self, global_step):
    self.H = 10 # number of hidden layer neurons
    self.D = 4 #input dimension
    self.learning_rate = 1e-4 #learning rate

    observations = tf.placeholder(tf.float32, [None, self.D + 1] , name="input_x")
    W1 = tf.get_variable("W1", shape=[self.D + 1, self.H],
                                    initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.matmul(observations, W1))
    W2 = tf.get_variable("W2", shape=[self.H, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(layer1, W2)
    probability = tf.nn.sigmoid(score)

    tvars = [W1, W2]
    advantages = tf.placeholder(tf.float32,name="reward_signal")

    loglik = tf.log((advantages - probability) + (1 - advantages)*(advantages + probability))
    loss = -tf.reduce_mean(loglik)

    tvars = [W1, W2]
    newGrads = tf.gradients(loss, tvars)

    adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
    W2Grad = tf.placeholder(tf.float32, name="batch_grad2")

    batchGrad = [W1Grad, W2Grad]
    updateGrads = adam.apply_gradients(zip(batchGrad, tvars), global_step=global_step)

    return {'W1':W1,
            'W2':W2,
            'loss':loss,
            'observations': observations,
            'probability': probability,
            'advantages': advantages,
            'newGrads': newGrads,
            'W1Grad': W1Grad,
            'updateGrads': updateGrads,
            'tvars': tvars,
            'W2Grad': W2Grad}

  # These variables will be saved by the saver
  def add_to_collection(self, results):
    tf.add_to_collection("W1", results["W1"])
    tf.add_to_collection("W2", results["W2"])
    tf.add_to_collection("observations", results["observations"])
    tf.add_to_collection("probability", results["probability"])
    tf.add_to_collection("advantages", results["advantages"])
    tf.add_to_collection("newGrads1", results["newGrads"][0])
    tf.add_to_collection("newGrads2", results["newGrads"][1])
    tf.add_to_collection("W1Grad", results["W1Grad"])
    tf.add_to_collection("updateGrads", results["updateGrads"])
    tf.add_to_collection("W2Grad", results["W2Grad"])

  # these values will be collected from the saver to restore a
  # trained model
  def get_collection(self, global_step):
    W1 = tf.get_collection("W1")[0]
    W2 = tf.get_collection("W2")[0]
    observations = tf.get_collection("observations")[0]
    probability = tf.get_collection("probability")[0]
    advantages = tf.get_collection("advantages")[0]
    W1Grad = tf.get_collection("W1Grad")[0]
    updateGrads = tf.get_collection("updateGrads")[0]
    W2Grad = tf.get_collection("W2Grad")[0]
    newGrads1 = tf.get_collection("newGrads1")[0]
    newGrads2 = tf.get_collection("newGrads2")[0]
    newGrads = [newGrads1, newGrads2]

    self.global_step = global_step
    self.W1 = W1
    self.W2 = W2
    self.observations = observations
    self.probability = probability
    self.advantages = advantages
    self.W1Grad = W1Grad
    self.updateGrads = updateGrads
    self.W2Grad = W2Grad
    self.newGrads = newGrads

  # Before training, any initialization code
  def before(self, sess):
    #store observations, actions and rewards
    self.xs, self.ys, self.rs = [],[],[]

  # Return an action
  def get_action(self, sess, observation):
    obs = np.hstack((observation, [[1]]))
    tfprob = sess.run(self.probability, feed_dict={self.observations: obs})
    action = 1 if np.random.uniform() > tfprob else 0

    self.xs.append(observation) # save observation
    self.ys.append(action)
    return action

  # After action has been processed by env, what to do with reward
  def after_action(self, sess, reward, info) :
    # just store the reward for later
    self.rs.append(reward)

  # After each episode
  def after_episode(self, sess):


    tGrad, global_step_val = sess.run([self.newGrads, self.global_step],
                                      feed_dict={self.observations: epx,
                                                 self.advantages: discounted_epr})


    sess.run(self.updateGrads, feed_dict={self.W1Grad: tGrad[0],
                                          self.W2Grad: tGrad[1]})

    return global_step_val

  # After each (bach size) episodes
  def after_batch(self, sess):
    # do nothing
    pass

  # After training
  def after(self):
    # do nothing
    pass

  def collect(self):
    pass

class PolicyGradient(BaseModel):

  @staticmethod
  def create_model():
    return PolicyGradient();

  # Modify the graph below, but add return values for the graph so we can save the model
  def build_graph(self, global_step):

    self.H = 10 # number of hidden layer neurons
    self.gamma = 0.99 # discount factor for reward
    self.D = 4 #input dimension
    self.learning_rate = 1e-2 #learning rate

    observations = tf.placeholder(tf.float32, [None, self.D] , name="input_x")
    W1 = tf.get_variable("W1", shape=[self.D, self.H],
                                    initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.matmul(observations, W1))
    W2 = tf.get_variable("W2", shape=[self.H, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(layer1, W2)
    probability = tf.nn.sigmoid(score)

    tvars = [W1, W2]
    input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
    advantages = tf.placeholder(tf.float32,name="reward_signal")

    loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
    loss = -tf.reduce_mean(loglik * advantages)
    newGrads = tf.gradients(loss, tvars)

    adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
    W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
    batchGrad = [W1Grad, W2Grad]
    updateGrads = adam.apply_gradients(zip(batchGrad, tvars), global_step=global_step)

    # return so we know what to keep
    return {'W1':W1,
            'W2':W2,
            'loss':loss,
            'observations': observations,
            'probability': probability,
            'input_y': input_y,
            'advantages': advantages,
            'newGrads': newGrads,
            'W1Grad': W1Grad,
            'updateGrads': updateGrads,
            'tvars': tvars,
            'W2Grad': W2Grad}


  # These variables will be saved by the saver
  def add_to_collection(self, results):
    tf.add_to_collection("W1", results["W1"])
    tf.add_to_collection("W2", results["W2"])
    tf.add_to_collection("observations", results["observations"])
    tf.add_to_collection("probability", results["probability"])
    tf.add_to_collection("input_y", results["input_y"])
    tf.add_to_collection("advantages", results["advantages"])
    tf.add_to_collection("newGrads1", results["newGrads"][0])
    tf.add_to_collection("newGrads2", results["newGrads"][1])
    tf.add_to_collection("W1Grad", results["W1Grad"])
    tf.add_to_collection("updateGrads", results["updateGrads"])
    tf.add_to_collection("W2Grad", results["W2Grad"])

  # these values will be collected from the saver to restore a
  # trained model
  def get_collection(self, global_step):
    W1 = tf.get_collection("W1")[0]
    W2 = tf.get_collection("W2")[0]
    observations = tf.get_collection("observations")[0]
    probability = tf.get_collection("probability")[0]
    input_y = tf.get_collection("input_y")[0]
    advantages = tf.get_collection("advantages")[0]
    W1Grad = tf.get_collection("W1Grad")[0]
    updateGrads = tf.get_collection("updateGrads")[0]
    W2Grad = tf.get_collection("W2Grad")[0]
    newGrads1 = tf.get_collection("newGrads1")[0]
    newGrads2 = tf.get_collection("newGrads2")[0]
    newGrads = [newGrads1, newGrads2]

    self.global_step = global_step
    self.W1 = W1
    self.W2 = W2
    self.observations = observations
    self.probability = probability
    self.input_y = input_y
    self.advantages = advantages
    self.W1Grad = W1Grad
    self.updateGrads = updateGrads
    self.W2Grad = W2Grad
    self.newGrads = newGrads


  # Before training, any initialization code
  def before(self, sess):
    #store observations, actions and rewards
    self.xs, self.ys, self.rs = [],[],[]

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run([self.W1, self.W2])
    for ix, grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0

    self.gradBuffer = gradBuffer

  # Return an action
  def get_action(self, sess, observation):
    tfprob = sess.run(self.probability, feed_dict={self.observations: observation})
    action = 1 if np.random.uniform() < tfprob else 0

    self.xs.append(observation) # save observation
    y = 1 if action == 0 else 0 # a "fake label"
    self.ys.append(y)
    return y

  # After action has been processed by env, what to do with reward
  def after_action(self, sess, reward, info) :
    # just store the reward for later
    self.rs.append(reward)

  # After each episode
  def after_episode(self, sess):
    # stack together all inputs, actions and rewards for this episode
    epx = np.vstack(self.xs)
    epy = np.vstack(self.ys)
    epr = np.vstack(self.rs)

    #clear episode variables
    self.xs, self.ys, self.rs = [],[],[]

    # compute the discounted reward backwards through time
    discounted_epr = PolicyGradient._discount_rewards(epr)

    # size the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    # get the gradients
    tGrad, global_step_val = sess.run([self.newGrads, self.global_step],
                                      feed_dict={self.observations: epx,
                                                 self.input_y: epy,
                                                 self.advantages: discounted_epr})

    # store gradients in grad buffer
    for ix,grad in enumerate(tGrad):
      self.gradBuffer[ix] += grad

    #return the global step value to the training harness
    return global_step_val

  # After each (bach size) episodes
  def after_batch(self, sess):

    # run updateGrads
    sess.run(self.updateGrads, feed_dict={self.W1Grad: self.gradBuffer[0],
                                          self.W2Grad: self.gradBuffer[1]})

    # clear the gradient buffer
    for ix,grad in enumerate(self.gradBuffer):
      self.gradBuffer[ix] = grad * 0


  # After training
  def after(self):
    # do nothing
    pass

  def collect(self):
    pass

  # private function for discounting rewards
  @staticmethod
  def _discount_rewards(r):
    gamma = .99
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r
