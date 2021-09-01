import pandas as pd
from math import exp

def EMA(data, timeperiod):
  ema_output = []
  sma = 0
  multiplier = 2 / (timeperiod + 1)
  for i in range(0, timeperiod):
    sma += data[i][4]
  sma /= timeperiod
  previous_ema = sma #initialize

  for i in range(timeperiod, len(data)):
    current_ema = data[i][4] * multiplier + previous_ema * (1 - multiplier)
    previous_ema = current_ema
    ema_output.append(current_ema)
  return ema_output

def MACD(data, short_ema, long_ema, timeperiod1, timeperiod2):
  macd_output = []
  t_diff = timeperiod2 - timeperiod1 - 1
  for i in range(0, len(long_ema)):
    macd = short_ema[i+t_diff] - long_ema[i]
    macd_output.append(macd)
  return macd_output

def RSI(data, timeperiod):
  rsi_output = []
  N = timeperiod + 1 
  moveU = 0 
  moveD = 0
  for i in range(1, N):
    change = data[i][4] - data[i-1][4]
    if change > 0: # up
      moveU += change
    elif change < 0: # down
      moveD -= change
    else:
      pass
    
  AvgU = moveU / timeperiod + 1
  AvgD = moveD / timeperiod + 1
  first_rsi = 100 - 100 / (1 + AvgU / AvgD)
  rsi_output.append(first_rsi)

  for i in range(15, len(data)):
    moveU = AvgU * 13
    moveD = AvgD * 13
    current = data[i][4] - data[i-1][4]
    if current > 0:
      moveU += current
    elif current < 0:
      moveD -= current
    else:
      pass
    AvgU = moveU / timeperiod + 1
    AvgD = moveD / timeperiod + 1
    current_rsi = 100 - 100 / (1 + AvgU / AvgD)
    rsi_output.append(current_rsi)

  return rsi_output

def ADL(data):
  adl_output = []
  init_multiplier = exp(((data[0][4] - data[0][3]) - (data[0][2] - data[0][4]))) -  exp((data[0][2] - data[0][3]))
  previous_ADL = init_multiplier * data[0][5]

  for i in range(1, len(data)):
    curr_data = data[i]
    current_multiplier = exp((curr_data[4] - curr_data[3]) - (curr_data[2] - curr_data[4])) - exp((curr_data[2] - curr_data[3]))
    current_volume = curr_data[5] * current_multiplier
    current_ADL = previous_ADL + current_volume
    previous_ADL = current_ADL
    adl_output.append(current_ADL)
  return adl_output

def refine(df):
  data = df.values.tolist()
  rsi_data = RSI(data, 14)
  ema12_data = EMA(data, 12)
  ema26_data = EMA(data, 26)
  macd_data = MACD(data, ema12_data, ema26_data, 12, 26)
  adl_data = ADL(data)

  dataset = [data, rsi_data, ema12_data, ema26_data, macd_data, adl_data]
  minimum = min(len(rsi_data), len(ema12_data), len(ema26_data), len(macd_data), len(adl_data))
  for i in range(0, len(dataset)):
    if len(dataset[i]) > minimum:
      mincut = len(dataset[i]) - minimum
      dataset[i] = dataset[i][mincut:]
  newdf = pd.DataFrame(dataset[0], columns =['open', 'high', 'low', 'close', 'volume', 'value'])
  newdf['rsi'] = dataset[1]
  newdf['ema12'] = dataset[2]
  newdf['ema26'] = dataset[3]
  newdf['macd'] = dataset[4]
  newdf['adl'] = dataset[5]
  for column in newdf.columns: # normalization
    newdf[column] = newdf[column] / newdf[column].abs().max()

  return newdf
