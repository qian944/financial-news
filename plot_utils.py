import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_stock_kline(df):
    # ... (这个函数保持不变，我们只是在它下面新增函数)
    if df is None or df.empty:
        return go.Figure()

    # ... (原来的K线图代码) ...
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=('K线图 & 移动平均线', '成交量'),
                        row_heights=[0.75, 0.25])
    # ... (K线, MA, 成交量) ...
    fig.add_trace(go.Candlestick(x=df['trade_date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'), row=1, col=1)
    ma5 = df['close'].rolling(window=5).mean()
    ma10 = df['close'].rolling(window=10).mean()
    ma20 = df['close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=df['trade_date'], y=ma5, mode='lines', name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=ma10, mode='lines', name='MA10', line=dict(color='purple', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=ma20, mode='lines', name='MA20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=df['trade_date'], y=df['vol'], name='成交量', marker_color='lightblue'), row=2, col=1)
    fig.update_layout(
        title_text=f"{df['ts_code'].iloc[0]} 近一个月走势分析", height=600,
        xaxis_title=None, xaxis2_title='交易日期', yaxis_title="价格", yaxis2_title="成交量",
        xaxis_rangeslider_visible=False, legend_title="图例"
    )
    return fig


# --- 新增蒙特卡洛模拟与绘图功能 ---

def run_monte_carlo_simulation(hist_df, sim_days=90, num_simulations=1000):
    """
    基于历史数据，运行蒙特卡洛模拟。
    
    Args:
        hist_df (pd.DataFrame): 包含'close'价格的历史数据DataFrame。
        sim_days (int): 模拟未来的天数。
        num_simulations (int): 模拟的次数。
        
    Returns:
        tuple: (模拟路径的DataFrame, 最终价格数组)
    """
    # 1. 计算历史日收益率
    log_returns = np.log(1 + hist_df['close'].pct_change())
    
    # 2. 计算漂移率(drift)和波动率(volatility)
    mu = log_returns.mean()
    var = log_returns.var()
    drift = mu - 0.5 * var
    stdev = log_returns.std()
    
    # 3. 生成随机变量 (几何布朗运动)
    daily_returns = np.exp(drift + stdev * np.random.standard_normal((sim_days, num_simulations)))
    
    # 4. 生成价格路径
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = hist_df['close'].iloc[-1] # 起始价格为最近一天的收盘价
    for t in range(1, sim_days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    # 5. 整理数据格式
    # 创建未来日期索引
    last_date = hist_df['trade_date'].iloc[-1]
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, sim_days + 1)])
    
    sim_df = pd.DataFrame(price_paths, index=future_dates)
    end_prices = sim_df.iloc[-1].values
    
    return sim_df, end_prices

def plot_monte_carlo(sim_df, end_prices, start_price):
    """
    绘制蒙特卡洛模拟的“意大利面图”和最终价格分布直方图。
    """
    # 创建带两个Y轴的子图，左边是路径图，右边是直方图
    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=True,
                        subplot_titles=('未来股价模拟路径', '最终价格分布'))

    # 1. 绘制所有模拟路径（意大利面）
    for col in sim_df.columns:
        fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df[col], mode='lines', 
                                 line=dict(color='grey', width=0.5), showlegend=False), 
                      row=1, col=1)

    # 2. 绘制中位数路径 (加粗蓝线)
    median_path = sim_df.median(axis=1)
    fig.add_trace(go.Scatter(x=sim_df.index, y=median_path, mode='lines',
                                 line=dict(color='blue', width=3), name='中位数路径'),
                      row=1, col=1)

    # 3. 绘制最终价格分布直方图 (在右边子图)
    fig.add_trace(go.Histogram(y=end_prices, name='价格分布', marker_color='#636EFA'), 
                  row=1, col=2)
    
    # 4. 美化布局
    sim_days = len(sim_df)
    fig.update_layout(
        title_text=f'未来 {sim_days} 天股价蒙特卡洛模拟 (1000次)',
        yaxis_title='股价',
        xaxis_title='日期'
    )
    
    # 计算概率结论
    prob_higher = np.mean(end_prices > start_price)
    
    return fig, prob_higher
