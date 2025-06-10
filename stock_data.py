import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

ts.set_token(st.secrets["TUSHARE_TOKEN"])
pro = ts.pro_api()

def get_stock_data(stock_code, end_date_str, days=365):
    """
    获取指定股票在某个结束日期前一段时间的日线数据。
    
    Args:
        stock_code (str): 股票代码, e.g., '000001.SZ'
        end_date_str (str): 结束日期字符串, e.g., '2024-06-10'
        days (int): 向前取数据的天数，默认为365天，用于计算统计量。
        
    Returns:
        pd.DataFrame or None: 返回包含日线数据的DataFrame，失败则返回None。
    """
    try:
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
        # 多取一些数据，比如 days + 100，确保有足够的交易日
        start_dt = end_dt - timedelta(days=days + 100) 
        
        start_date_tushare = start_dt.strftime('%Y%m%d')
        end_date_tushare = end_dt.strftime('%Y%m%d')

        df = ts.pro_bar(ts_code=stock_code,
                        start_date=start_date_tushare,
                        end_date=end_date_tushare,
                        asset='E',
                        freq='D')

        if df is None or df.empty:
            return None

        # 数据预处理
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values("trade_date", ascending=True)
        # 只保留最近的 N 个交易日的数据，保证数据量
        df = df.tail(days) 
        
        return df

    except Exception as e:
        st.error(f"Tushare 数据获取异常: {e}")
        return None
