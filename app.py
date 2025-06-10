import streamlit as st
from predict_model import predict_by_model
from predict_ai import predict_by_ai, check_timeliness, generate_investment_advice
from stock_data import get_stock_data
from plot_utils import plot_stock

st.title("📈 财经新闻智能判别与投资建议系统")

# 输入部分
title = st.text_input("新闻标题")
content = st.text_area("新闻正文")
platform = st.selectbox("新闻发布平台", ["官方媒体", "财经媒体/商业媒体", "社交平台/自媒体"])
platform_code = {"官方媒体": 0, "财经媒体/商业媒体": 1, "社交平台/自媒体": 2}[platform]
date = st.date_input("发布日期(yyyy/mm/dd)")

# 判别方式选择
mode = st.radio("请选择判别方式", ["使用模型", "使用AI"])

# 开始判别按钮
if st.button("开始判别"):
    if mode == "使用模型":
        result, prob = predict_by_model(title, content, platform_code)
        st.success(f"模型判定结果：{result}（真实概率：{prob[1]:.2f}）")
    else:
        result = predict_by_ai(title, content, platform_code)
        st.success(f"AI 判定结果：{result}")

    # 投资建议模块
    if st.checkbox("需要投资建议", value=True):
        stock_code = st.text_input("请输入股票代码（如000001.SZ）")
        if stock_code and st.button("生成投资建议"):
            if check_timeliness(str(date), stock_code):
                stock_df = get_stock_data(stock_code, str(date))
                if stock_df is not None:
                    st.pyplot(plot_stock(stock_df))
                    suggestion = generate_investment_advice(f"{title} {content}", stock_df.to_dict())
                    st.markdown(suggestion)
                else:
                    st.error("股票数据获取失败")
            else:
                st.warning("此信息或已失效，请您谨慎投资")
