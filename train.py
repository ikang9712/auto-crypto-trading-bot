import numpy as np
import tensorflow as tf
import random
import sys
from datetime import datetime
from env import SingleQEnv

CHOICES = list(range(0, 101))
EPISODE_REWARDS = []
ACTION_SPACE_SIZE = 101 
STATE_SPACE_SIZE = 13 
SHOW_EVERY = 1000

class LinearModel:
  def __init__(self, lr, gamma, weights, bias):
    self.lr = lr
    self.gamma = gamma
    self.weights = weights
    self.bias = bias

  def predict(self, state):
    z = state@self.weights
    return z
  
  def get_gradient(self, state, action):
    weights_T = np.zeros([ACTION_SPACE_SIZE, STATE_SPACE_SIZE])
    weights_T[action] = state + self.bias
    return weights_T.T

  def update(self, curr_s, curr_opt_a, next_s, next_opt_a, r, switch = False):
    curr_q_values = self.predict(curr_s)
    curr_q = curr_q_values[0, curr_opt_a].item()
    next_q_values = self.predict(next_s)
    next_q = next_q_values[0, next_opt_a].item()
    weight_gradient = self.get_gradient(curr_s, curr_opt_a)
    if switch:
      print(f"weights:{self.weights}")
      print(f"curr_q: {curr_q}")
      print(f"next_q: {next_q}")

    self.weights -= self.lr * (curr_q - (r + self.gamma * next_q)) * weight_gradient
    self.bias -= self.lr * (curr_q - (r + self.gamma * next_q))

class AutoTradingAgent:
  def __init__(self, env, gamma, lr, epsilon, epsilon_decay, initial_weights, initial_bias):
    self.env = env
    self.gamma = gamma
    self.lr = lr
    self.epsilon = epsilon 
    self.epsilon_decay = epsilon_decay
    self.initial_weights = initial_weights
    self.initial_bias = initial_bias
    self.lm = None

  def get_action(self, state):
    q_values = self.lm.predict(state)
    opt_action = np.argmax(q_values)
    p = np.random.random()
    self.epsilon *= self.epsilon_decay
    if p > self.epsilon:
      random_action = np.random.randint(low = 30, high = 71) 
      # restricts the exploration in the range of [30, 70]
      return random_action
    else:
      return opt_action
  
  def train(self, episodes, max_iterations):
    self.lm = LinearModel(self.lr, self.gamma, self.initial_weights, self.initial_bias)
    for episode in range(episodes):
      # LOCAL REPORT
      if episode % SHOW_EVERY == 0 and episode > 0:
        print(f"on # {episode}, epsilon: {self.epsilon}")
        print(f"mean episoode reward: {np.mean(EPISODE_REWARDS)}")
      
      # RESET SETUP
      curr_state = self.env.reset(max_iterations)
      local_reward = 0
      start = self.env.seed
      for i in range(start, max_iterations):
        opt_action = self.get_action(curr_state)
        next_state, r = self.env.step(opt_action, curr_state)
        next_opt_action = self.get_action(next_state)
        if episode < 1 and i < 50:
          print(f"iteration: {i}")
          self.lm.update(curr_state, opt_action, next_state, next_opt_action, r, True)
        else:
          self.lm.update(curr_state, opt_action, next_state, next_opt_action, r, False)
        local_reward += r
        curr_state = next_state
      EPISODE_REWARDS.append(local_reward)
    return self.lm.weights, self.lm.bias
  
def report(path, w, b):
  w_path = path + '_weights.txt'
  b_path = path + '_bias.txt'
  r_path = path + '_rewards.txt'
  np.savetxt(w_path, w, delimiter=',')
  np.savetxt(b_path, [b], delimiter = ',')
  np.savetxt(r_path, EPISODE_REWARDS, delimiter = '\n')

def load(w_outpath, b_outpath):
  if w_outpath == None:
    weights = np.random.uniform(low=0.1, high=1.0, size=(STATE_SPACE_SIZE,ACTION_SPACE_SIZE))
  else:
    weights = np.loadtxt(w_outpath, delimiter=',')
  if b_outpath == None:
    bias = 0.0
  else:
    bias = np.loadtxt(b_outpath, delimiter = ',')
  return weights, bias

def main(args):
  # args
  discount = float(args[1]) if len(args)>1 else 0.999 
  num_episodes = int(args[2]) if len(args)>2 else 25000
  epsilon = float(args[3]) if len(args)>3 else 0.9
  epsilon_decay = float(args[4]) if len(args)>4 else 0.98
  learning_rate = float(args[5]) if len(args)>5 else 0.01
  train_data_size = int(args[6]) if len(args)>6 else 950000
  weights_inpath = args[7] if len(args)>7 else None # example) "./s.txt"
  bias_inpath = args[8] if len(args)>8 else None # example) "./bias2.txt"
  now = datetime.now()
  now = now.strftime("%H:%M")
  outpath_name = args[9] if len(args)>9 else f"result({now})"
  # setup
  env = SingleQEnv(ACTION_SPACE_SIZE, STATE_SPACE_SIZE)
  weights, bias = load(weights_inpath, bias_inpath)
  agent = AutoTradingAgent(env, discount, learning_rate, epsilon, epsilon_decay, weights, bias)
  final_weights, final_bias = agent.train(num_episodes, train_data_size)
  report(outpath_name, final_weights, final_bias)

if __name__ == "__main__":
  main(sys.argv)