import sys
sys.path.insert(0, "/home/leo/lob_research")

import pandas as pd
from src.ingestion.coinbase import process

events = process("/home/leo/lob_research/data/raw/coinbase/20260323_BTC-USDT.csv")

dups = events[events.duplicated(subset=["order_id", "session_id", "event_seq"], keep=False)]
print(dups.sort_values(["order_id", "session_id", "event_seq"]).head(20).to_string())