import urllib2
import logging
import sys
import json
import numpy as np
import time
import urllib
class Service(object):

  SERVER = "https://mlcc-158008.appspot.com"

  def __init__(self):
    #connect to server to make
    make_url = self.SERVER + "/make?r=" + str(time.time())
    response = urllib2.urlopen(make_url)
    self.cookie = response.headers.get('Set-Cookie')
    if (response.getcode() != 200) :
      raise Exception("cant connect to %s" % make_url)
    html = response.read()

  def reset(self):
    reset_url = self.SERVER + "/reset?r="+ str(time.time())
    request = urllib2.Request(reset_url)
    request.add_header('cookie', self.cookie)
    response = urllib2.urlopen(request)
    if (response.getcode() != 200) :
      raise Exception("cant connect to %s" % reset_url)

    html = response.read()
    obs = json.loads(html)
    return np.array(obs)

  def step(self, action):
    step_url = self.SERVER + "/step?action=%d&r=%s" % (action, str(time.time()))
    sys.stdout.write('.')
    sys.stdout.flush()
    request = urllib2.Request(step_url)
    request.add_header('cookie', self.cookie)
    response = urllib2.urlopen(request)
    if (response.getcode() != 200) :
      raise Exception("cant connect to %s" % step_url)

    html = response.read()
    obs = json.loads(html)
    return [np.array(obs[0]), obs[1], obs[2], obs[3]]


  def submit(self, user, password):
    submit_url = self.SERVER + "/submit?r=%s" % str(time.time())
    kaggle_auth = {"user": user, "password": password}
    data = json.dumps(kaggle_auth)
    data_len = len(data)
    request = urllib2.Request(submit_url, data,
                              {'Content-Type': 'application/json',
                               'Content-Length': data_len})
    request.add_header('cookie', self.cookie)
    response = urllib2.urlopen(request)
    if (response.getcode() != 200) :
      raise Exception("cant connect to %s" % submit_url)
    return response.read()
