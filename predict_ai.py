import requests
import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def gemini_prompt(message: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": message}]
            }
        ]
    }

    response = requests.post(GEMINI_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"❌ Gemini请求失败: {response.status_code}\n{response.text}"

def predict_by_ai(title, content, platform_code):
    prompt = f"""以下是财经新闻信息，请判断其是否真实，并简要解释理由：
标题：{title}
内容：{content}
发布平台类型（0=官方，1=财经媒体/商业媒体，2=社交平台/自媒体）：{platform_code}
请仅返回“真实”或“虚假”+理由。"""
    return gemini_prompt(prompt)

from datetime import date

def check_timeliness(news_date, stock_code):
    today = date.today().strftime("%Y-%m-%d")
    prompt = f"""今天是 {today}。
请判断以下财经新闻是否仍具有投资参考价值。
发布日期为 {news_date}，相关股票代码为 {stock_code}。
回答“仍具有时效性”或“此信息或已失效”。"""
    return "失效" not in gemini_prompt(prompt)

def generate_investment_advice(news, stock_data_dict):
    prompt = f"""根据以下财经新闻与股票数据，请以以下格式输出投资建议：
【新闻概述】：一句话总结新闻内容  
【股票走势分析】：说明近一周涨跌趋势  
【投资建议】：明确买入/观望/卖出并给出理由
新闻：{news}  
数据：{stock_data_dict}
请严格按上述格式返回。"""
    return gemini_prompt(prompt)
