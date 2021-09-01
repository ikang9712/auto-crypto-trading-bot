from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import sqlite3

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()
# Parameters
currency = 'BTC'
initial_cash_balance = 100000.0 # won
initial_stock_balance = 0.0
data_path = '/Users/inhokang/Desktop/crypto/AutoCryptoTrading/crypto_data.db'

# Env
class CryptoEnv(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=100, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(13,), dtype=np.float32, minimum=0, name='observation')
    self._state = None
    self._data = []
    self._data_index = 1
    self._episode_ended = False
    self._local_reward = 0.0

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    if len(self._data) == 0:
      con = sqlite3.connect(data_path)
      cur = con.cursor()
      for row in cur.execute(f'SELECT * FROM {currency}'):
        self._data.append(row)
      con.close()
    self._state = [initial_cash_balance, initial_stock_balance] + list(self._data[0])
    self._episode_ended = False
    self._data_index = 1
    return ts.restart(np.array(self._state, dtype=np.float32))

  def _step(self, action):
    # Make sure episodes don't go on forever.
    if self._data_index >= len(self._data):
      self._episode_ended = True
    
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    curr_total_asset = self._state[0] + self._state[1] * self._state[5]
    if action in range(0,101):
      next_cash = curr_total_asset * (100 - action) / 100
      next_stock = curr_total_asset * action / 100 / self._state[5]
      next_state = [next_cash, next_stock] + list(self._data[self._data_index])
      next_total_asset = next_cash + next_stock * next_state[5]
      self._local_reward = next_total_asset - curr_total_asset
      
      self._state = next_state
    else:
      raise ValueError('`action` should be in range of 0 and 100.')

    # terminate or transit
    if self._episode_ended or curr_total_asset < 10000:
      return ts.termination(np.array(self._state, dtype = np.float32), self._local_reward)
    else:
      self._data_index += 1
      return ts.transition(np.array(self._state, dtype = np.float32), self._local_reward, discount=0.9999)


environment = CryptoEnv()
utils.validate_py_environment(environment, episodes=1)