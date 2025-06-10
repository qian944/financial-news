import matplotlib.pyplot as plt

def plot_stock(df):
    fig, ax = plt.subplots()
    ax.plot(df['trade_date'], df['close'], marker='o')
    ax.set_title("近一周股价")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价")
    plt.xticks(rotation=45)
    return fig
