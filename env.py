###############################DATAPATH######################################
###############################DATAPATH######################################
###############################DATAPATH######################################
DATAPATH = '/Users/inhokang/Desktop/crypto/AutoCryptoTrading/crypto_data.db'
###############################DATAPATH######################################
###############################DATAPATH######################################
###############################DATAPATH######################################
import sys
import numpy as np
import sqlite3
import random

class SingleQEnv:
  def __init__(self, action_size = None, state_size = None):
    self.action_size = action_size
    self.state_size = state_size
    self.data = []
    self.stock_name = None
    self.initial_cash_balance = 0.0
    self.initial_stock_balance = 0.0
    self.seed = 0

  def data_load(self):
    if len(self.data) == 0:
      con = sqlite3.connect(DATAPATH)
      cur = con.cursor()
      for row in cur.execute(f'SELECT * FROM {self.stock_name}'):
        self.data.append(row)
      con.close()

  def reset(self, seed_limit):
    with open('initial_cash.txt', 'r') as cash:
      contents = cash.readlines()
      self.initial_cash_balance = int(contents[0][:-1])
      self.stock_name = contents[1]
    if self.initial_cash_balance == 0 or self.stock_name == None:
      raise ValueError("error in initial cash and stock setup")
    self.data_load()
    self.seed = random.randrange(0,seed_limit // 10, seed_limit // 100)
    status = list(self.data[self.seed])
    status.insert(0, self.initial_stock_balance)
    status.insert(0, self.initial_cash_balance)
    initial_state = np.asarray(status).reshape((1,self.state_size))
    return initial_state
  
  def step(self, action, curr_state): # action = the cash ratio out of the total asset(cash + stock)
    next_state = list(self.data[self.seed])
    price_diff = random.randrange(-10000, 11000, 1000) # for possible latency
    # current state 
    current_sum = curr_state[0,0] + curr_state[0,1] * curr_state[0,5]
    current_cash_p = curr_state[0,0] / current_sum
    gap = int(action - current_cash_p)
    amount = gap * current_sum  / 100
    stock_price = curr_state[0,5] + price_diff
    # action
    if amount == 0: # do nothing
      next_cash = curr_state[0,0]
      next_stock = curr_state[0,1]
    else: # sell or buy stocks
      next_cash = curr_state[0,0] + amount
      next_stock = curr_state[0,1] - amount / stock_price
    # next state
    next_state.insert(0, next_stock)
    next_state.insert(0, next_cash)
    next_sum = next_state[0] + next_state[1] * next_state[5]
    # calculate rewards
    r = next_sum - current_sum
    next_state = np.asarray(next_state).reshape((1,self.state_size))
    return next_state, r

  def render(self, mode = 'human'):
    pass
