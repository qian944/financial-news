import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_stock_kline(df):
    """
    使用 Plotly 绘制包含K线、交易量和移动平均线的股票图表。
    
    Args:
        df (pd.DataFrame): 包含'trade_date', 'open', 'close', 'high', 'low', 'vol'的DataFrame。
        
    Returns:
        go.Figure: 返回一个Plotly Figure对象。
    """
    if df is None or df.empty:
        return go.Figure()

    # 创建一个带两个y轴的图表，上面是K线，下面是交易量
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('K线图 & 移动平均线', '成交量'),
                        row_width=[0.2, 0.7])

    # 1. 添加K线图 (蜡烛图)
    fig.add_trace(go.Candlestick(x=df['trade_date'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='K线'),
                  row=1, col=1)

    # 2. 计算并添加移动平均线 (MA)
    ma5 = df['close'].rolling(window=5).mean()
    ma10 = df['close'].rolling(window=10).mean()
    ma20 = df['close'].rolling(window=20).mean()

    fig.add_trace(go.Scatter(x=df['trade_date'], y=ma5, mode='lines', name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=ma10, mode='lines', name='MA10', line=dict(color='purple', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['trade_date'], y=ma20, mode='lines', name='MA20', line=dict(color='blue', width=1)), row=1, col=1)

    # 3. 添加成交量图
    fig.add_trace(go.Bar(x=df['trade_date'], y=df['vol'], name='成交量', marker_color='lightblue'),
                  row=2, col=1)
    
    # 4. 美化图表布局
    fig.update_layout(
        title_text=f"{df['ts_code'].iloc[0]} 近一个月走势分析",
        height=600, # 可以适当增加图表总高度
        
        # --- 核心改动：分别控制每个轴的标题 ---
        xaxis_title=None,  # 隐藏中间的X轴标题
        xaxis2_title='交易日期', # 只在最下面的X轴(xaxis2)显示标题
        
        yaxis_title="价格",
        yaxis2_title="成交量",
        
        xaxis_rangeslider_visible=False,
        legend_title="图例"
    )
    return fig
