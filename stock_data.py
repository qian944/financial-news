import tushare as ts
import pandas as pd
import streamlit as st

ts.set_token(st.secrets["TUSHARE_TOKEN"])
pro = ts.pro_api()

def get_stock_data(stock_code, pub_date):
    start = pd.to_datetime(pub_date) - pd.Timedelta(days=3)
    end = pd.to_datetime(pub_date) + pd.Timedelta(days=3)
    try:
        df = ts.pro_bar(ts_code=stock_code,
                        start_date=start.strftime('%Y%m%d'),
                        end_date=end.strftime('%Y%m%d'))
        return df.sort_values("trade_date") if df is not None else None
    except:
        return None
