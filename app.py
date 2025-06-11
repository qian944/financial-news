import streamlit as st
import numpy as np
from predict_model import predict_by_model
from predict_ai import predict_by_ai, check_timeliness, generate_investment_advice
from stock_data import get_stock_data
from plot_utils import plot_stock_kline
from plot_utils import plot_stock_kline, run_monte_carlo_simulation, plot_monte_carlo
from predict_ai import generate_monte_carlo_advice

# --- 第1步：初始化 Session State ---
# 把所有可能在多次刷新中需要保持状态的变量都在这里初始化
if 'show_results' not in st.session_state:
    st.session_state.show_results = False  # 控制是否显示结果区域
if 'result_data' not in st.session_state:
    st.session_state.result_data = {}  # 存储判别结果
if 'stock_code_input' not in st.session_state:
    st.session_state.stock_code_input = "" # 存储股票代码输入
if 'show_investment_analysis' not in st.session_state:
    st.session_state.show_investment_analysis = False # 控制整个分析模块的显示
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {} # 存储分析所需的所有数据和结果
if 'sim_days' not in st.session_state:
    st.session_state.sim_days = 90 # 默认模拟90天


st.markdown("""
<h2 style='text-align: center;'>
    📈 财经新闻智能判别与投资建议系统
</h2>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-family: "Georgia", serif; font-style: italic; font-size: 18px;'>
    这个消息可信吗？
</p>
""", unsafe_allow_html=True)

with st.container(border=True):
    # 使用 markdown 和 HTML h3 标签，并添加一个漂亮的下边框
    st.markdown("""
    <h3 style='border-bottom: 2px solid #D5DBDB; padding-bottom: 5px;'>
        第一步：输入新闻信息
    </h3>
    """, unsafe_allow_html=True)

    title = st.text_input("新闻标题")
    content = st.text_area("新闻正文", height=200)

    col1, col2 = st.columns([2, 3])
    with col1:
        platform = st.selectbox(
            "新闻发布平台",
            ["官方媒体", "财经媒体/商业媒体", "社交平台/自媒体"],
            key='platform_select'
        )
        platform_code = {"官方媒体": 0, "财经媒体/商业媒体": 1, "社交平台/自媒体": 2}[platform]
    with col2:
        st.write("")
        st.write("")
        with st.expander("查看不同平台的类型示例"):
            try:
                with open("来源分类 .txt", "r", encoding="utf-8") as f:
                    example_text = f.read()
                st.markdown(example_text)
            except FileNotFoundError:
                st.error("示例文件 '来源分类.txt' 未找到。")
    
    date = st.date_input("发布日期")
    mode = st.radio("请选择判别方式", ["使用我们的模型！", "使用AI(Gemini 2.0-flash)"], horizontal=True) 
# --- 输入部分保持不变 ---


# --- 第2步：处理“开始判别”按钮的点击事件 ---
if st.button("开始判别", type="primary"):
    if not title or not content:
        st.warning("新闻标题和正文不能为空！")
    else:
        st.session_state.show_results = True  # 只要点击了，就设置状态为“显示结果”
        st.session_state.result_data = {} # 清空旧数据
        with st.spinner("正在智能判别中..."):
            if mode == "使用我们的模型！":
                result, prob, sentiment = predict_by_model(title, content, platform_code)
                # 把结果存入session_state
                st.session_state.result_data = {
                    "mode": "model", "result": result, "prob": prob, "sentiment": sentiment
                }
            else: # 使用AI
                result = predict_by_ai(title, content, platform_code)
                st.session_state.result_data = {
                    "mode": "ai", "result": result
                }
        # 当按钮点击后，Streamlit会重新运行脚本，下面的代码会在刷新后执行

# --- 第3步：根据 Session State 的状态来决定是否显示结果区域 ---
if st.session_state.show_results:
    st.divider()
    
    # 同样使用自定义的 markdown 标题
    st.markdown("""
    <h3 style='border-bottom: 2px solid #D5DBDB; padding-bottom: 5px;'>
        第二步：查看判别结果与获取投资建议
    </h3>
    """, unsafe_allow_html=True)
    
    data = st.session_state.result_data

    # 显示判别结果
    if data.get("mode") == "model":
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"模型判定结果：**{data['result']}**")
            st.write(f"判定为真实的概率: **{data['prob'][1]:.2%}**")
        with col2:
            st.metric(label="文章情感分析得分", value=f"{data['sentiment']:.4f}")
            if data['sentiment'] > 0.02:
                st.info("情感倾向：偏向积极/乐观 😊")
            elif data['sentiment'] < -0.02:
                st.warning("情感倾向：偏向消极/悲观 😟")
            else:
                st.info("情感倾向：中性 😐")
    elif data.get("mode") == "ai":
        st.success(f"AI 判定结果：{data['result']}")
    
    # --- 投资建议模块现在被包裹在结果区域内，可以持久显示 ---
    if st.checkbox("需要投资建议与分析", value=True):
        st.session_state.stock_code_input = st.text_input(
            "请输入股票代码（如000001.SZ）", 
            value=st.session_state.stock_code_input
        )

        # "生成分析"按钮现在只负责数据获取和设置显示状态
        if st.button("生成投资建议与分析", use_container_width=True):
            if st.session_state.stock_code_input:
                st.session_state.show_investment_analysis = True # 打开显示开关
                st.session_state.analysis_data = {} # 清空旧数据
                
                with st.spinner("正在获取长短期数据..."):
                    stock_code = st.session_state.stock_code_input
                    news_date_str = str(date.strftime('%Y-%m-%d'))
                    
                    if check_timeliness(news_date_str, stock_code):
                        hist_df = get_stock_data(stock_code, news_date_str, days=365)
                        # 将获取到的数据存入 state
                        st.session_state.analysis_data['hist_df'] = hist_df
                        st.session_state.analysis_data['news_title'] = title
                        st.session_state.analysis_data['news_content'] = content
                    else:
                        st.warning("此信息或已失效，请您谨慎投资")
                        st.session_state.show_investment_analysis = False # 如果失效，就不显示
            else:
                st.warning("请输入股票代码！")
        
        # --- 所有显示逻辑，都由 session_state 控制，而不是 button 控制 ---
        if st.session_state.show_investment_analysis and 'hist_df' in st.session_state.analysis_data:
            
            hist_df = st.session_state.analysis_data.get('hist_df')

            if hist_df is not None and not hist_df.empty:
                
                # --- Part 1: 基本面与技术面分析 ---
                st.subheader("Part 1: 基于新闻的基本面与技术面分析")
                recent_df = hist_df.tail(30)
                st.plotly_chart(plot_stock_kline(recent_df), use_container_width=True)
                
                # 从 state 中恢复新闻内容
                news_title = st.session_state.analysis_data.get('news_title', '')
                news_content = st.session_state.analysis_data.get('news_content', '')

                short_data_str = hist_df.tail(7)[['trade_date', 'open', 'close', 'high', 'low', 'vol']].to_string()
                long_data_str = recent_df[['trade_date', 'open', 'close', 'high', 'low', 'vol']].to_string()
                
                suggestion = generate_investment_advice(f"{news_title} {news_content}", short_data_str, long_data_str)
                st.markdown(suggestion)
                
                st.divider()

                # --- Part 2: 蒙特卡洛模拟 ---
                st.subheader("Part 2: 基于蒙特卡洛模拟的未来股价概率分析")
                
                # 将 selectbox 的值与 session_state 绑定
                st.session_state.sim_days = st.selectbox(
                    "选择模拟周期（天）", 
                    [30, 90, 365], 
                    index=[30, 90, 365].index(st.session_state.sim_days) # 保证选择的值被记住
                )
                
                with st.spinner(f"正在进行 {st.session_state.sim_days} 天的蒙特卡洛模拟..."):
                    sim_df, end_prices = run_monte_carlo_simulation(hist_df, sim_days=st.session_state.sim_days, num_simulations=1000)
                    
                    start_price = hist_df['close'].iloc[-1]
                    mc_fig, prob_higher = plot_monte_carlo(sim_df, end_prices, start_price)
                    
                    st.plotly_chart(mc_fig, use_container_width=True)
                    
                    median_price = np.median(end_prices)
                    mc_advice = generate_monte_carlo_advice(st.session_state.stock_code_input, st.session_state.sim_days, prob_higher, median_price)
                    st.markdown(mc_advice)
            else:
                st.error("股票数据获取失败，请检查代码或日期。")
