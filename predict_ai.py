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
    """
    检查新闻时效性。
    通过设定明确的规则和输出格式，引导AI做出更符合预期的判断。
    """
    today_str = date.today().strftime('%Y-%m-%d')
    
    prompt = f"""
    你是一个金融分析助手。请根据以下规则判断新闻的时效性。

    **规则：**
    1.  **短期事件驱动型新闻**（如：公司发布财报、高管变动、突发公告）：如果新闻发布日期距离今天（{today_str}）超过7天，则认为时效性较低。
    2.  **长期趋势/基本面新闻**（如：行业分析、公司战略、研发进展）：如果新闻发布日期距离今天超过30天，则认为时效性较低。
    3.  请综合考虑新闻内容和日期，做出判断。

    **任务：**
    新闻发布日期：{news_date}
    涉及股票：{stock_code}

    请分析以上信息，并从以下两个选项中选择一个作为你的回答，不要添加任何额外的解释或文字。

    **选项：**
    - [A] 具有参考价值
    - [B] 时效性较低，谨慎参考

    请直接返回你的选择，例如：[A]
    """
    
    response = gemini_prompt(prompt)
    
    # 打印AI的原始回复，方便调试
    print(f"Gemini 时效性判断原始回复: {response}")
    
    # 检查回复中是否包含 "[A]" 或 "参考价值"
    # 只要不明确说 "时效性较低" 或 "[B]"，我们都默认为有效
    if "[B]" in response or "较低" in response:
        return False
    else:
        return True

def generate_investment_advice(news, stock_data_dict):
    prompt = f"""以下是新闻与近一周股票数据，请用以下格式生成投资建议：
【新闻概述】：  
【股票走势分析】：  
【投资建议】：  
新闻：{news}  
数据：{stock_data_dict}"""
    return gemini_prompt(prompt)
