import requests
import streamlit as st
from datetime import date

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def gemini_prompt(message: str):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": message}]}]
    }
    response = requests.post(GEMINI_URL, json=payload, headers=headers)
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "❌ Gemini响应异常"

def predict_by_ai(title, content, platform_code):
    prompt = f"""请判断以下财经新闻是否真实，并简要说明理由：
标题：{title}
内容：{content}
平台类型（0=官方，1=财经媒体/商业媒体，2=社交平台/自媒体）：{platform_code}
请仅返回“真实”或“虚假”+简短理由。"""
    return gemini_prompt(prompt)

def check_timeliness(news_date, stock_code):
    prompt = f"""今天是 {date.today().strftime('%Y-%m-%d')}。
请判断发布日期为 {news_date} 且涉及股票 {stock_code} 的新闻是否仍具有投资时效。
请仅返回“仍具有时效性”或“此信息或已失效”。"""
    return "失效" not in gemini_prompt(prompt)

def generate_investment_advice(news, stock_data_dict):
    prompt = f"""以下是新闻与近一周股票数据，请用以下格式生成投资建议：
【新闻概述】：  
【股票走势分析】：  
【投资建议】：  
新闻：{news}  
数据：{stock_data_dict}"""
    return gemini_prompt(prompt)
