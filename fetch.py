###############################DATAPATH######################################
###############################DATAPATH######################################
###############################DATAPATH######################################
DATAPATH = '/Users/inhokang/Desktop/crypto/AutoCryptoTrading/crypto_data.db'
###############################DATAPATH######################################
###############################DATAPATH######################################
###############################DATAPATH######################################
import pyupbit
import pandas as pd
import sqlite3
import sys
from furnish import refine

def fetch(crypto_name, data_size, time):
  df = pyupbit.get_ohlcv(f"KRW-{crypto_name}", count=data_size,interval=time)
  final_df = refine(df)
  con = sqlite3.connect(DATAPATH)
  final_df.to_sql(f"{crypto_name}", con, index = False, if_exists = "replace")
  con.close()

def main(args):
  crypto_name = args[1] if len(args)>1 else "BTC"
  data_size = int(args[2]) if len(args)>2 else 1000000
  interval = args[3] if len(args)>3  else "minute1"
  fetch(crypto_name, data_size, interval)

if __name__ == "__main__":
  main(sys.argv)