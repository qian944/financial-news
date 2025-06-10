import streamlit as st
from predict_model import predict_by_model
from predict_ai import predict_by_ai, check_timeliness, generate_investment_advice
from stock_data import get_stock_data
from plot_utils import plot_stock

# --- 第1步：初始化 Session State ---
# 把所有可能在多次刷新中需要保持状态的变量都在这里初始化
if 'show_results' not in st.session_state:
    st.session_state.show_results = False  # 控制是否显示结果区域
if 'result_data' not in st.session_state:
    st.session_state.result_data = {}  # 存储判别结果
if 'stock_code_input' not in st.session_state:
    st.session_state.stock_code_input = "" # 存储股票代码输入


st.title("📈 财经新闻智能判别与投资建议系统")

# --- 输入部分保持不变 ---
with st.container(border=True):
    st.subheader("第一步：输入新闻信息")
    title = st.text_input("新闻标题")
    content = st.text_area("新闻正文", height=200)
    platform = st.selectbox("新闻发布平台", ["官方媒体", "财经媒体/商业媒体", "社交平台/自媒体"])
    platform_code = {"官方媒体": 0, "财经媒体/商业媒体": 1, "社交平台/自媒体": 2}[platform]
    date = st.date_input("发布日期(yyyy/mm/dd)")
    mode = st.radio("请选择判别方式", ["使用我们的模型！", "使用AI"])

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
    st.subheader("第二步：查看判别结果与获取投资建议")
    
    data = st.session_state.result_data

    # 显示判别结果
    if data.get("mode") == "model":
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"模型判定结果：**{data['result']}**")
            st.write(f"判定为真实的概率: **{data['prob'][1]:.2%}**")
        with col2:
            st.metric(label="文章情感分析得分", value=f"{data['sentiment']:.4f}")
            if data['sentiment'] > 0.01:
                st.info("情感倾向：偏向积极/乐观 😊")
            elif data['sentiment'] < -0.01:
                st.warning("情感倾向：偏向消极/悲观 😟")
            else:
                st.info("情感倾向：中性 😐")
    elif data.get("mode") == "ai":
        st.success(f"AI 判定结果：{data['result']}")
    
    # --- 投资建议模块现在被包裹在结果区域内，可以持久显示 ---
    if st.checkbox("需要投资建议", value=True):
        # 将输入框的值绑定到 session_state
        st.session_state.stock_code_input = st.text_input(
            "请输入股票代码（如000001.SZ）", 
            value=st.session_state.stock_code_input
        )

        if st.button("生成投资建议"):
            if st.session_state.stock_code_input:
                with st.spinner("正在获取数据并生成建议..."):
                    if check_timeliness(str(date), st.session_state.stock_code_input):
                        stock_df = get_stock_data(st.session_state.stock_code_input, str(date))
                        if stock_df is not None:
                            # 为了防止图表和建议在下次输入时还显示，可以考虑也用session_state存
                            st.pyplot(plot_stock(stock_df))
                            suggestion = generate_investment_advice(f"{title} {content}", stock_df.to_dict())
                            st.markdown(suggestion)
                        else:
                            st.error("股票数据获取失败")
                    else:
                        st.warning("此信息或已失效，请您谨慎投资")
            else:
                st.warning("请输入股票代码！")
