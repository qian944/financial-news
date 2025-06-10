import tushare as ts
import pandas as pd
from settings import TUSHARE_TOKEN

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def get_stock_data(stock_code, pub_date):
    start = pd.to_datetime(pub_date) - pd.Timedelta(days=3)
    end = pd.to_datetime(pub_date) + pd.Timedelta(days=3)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    try:
        hist = ts.pro_bar(ts_code=stock_code, start_date=start_str, end_date=end_str)
        return hist.sort_values('trade_date')
    except Exception:
        return None
