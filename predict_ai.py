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

def generate_investment_advice(news, stock_data_short_str, stock_data_long_str):
    """
    根据新闻、短期（7天）和长期（30天）的股票数据生成投资建议。
    """
    prompt = f"""
    你是一位资深的金融市场分析师。请基于以下新闻事件、短期（过去7个交易日）和长期（过去30个交易日）的股票数据，提供一份专业的投资建议。

    **分析要求：**
    1.  **新闻解读**：简要概括新闻的核心内容，并分析它可能对股价产生的短期和长期影响。
    2.  **技术面分析**：
        - 结合**短期数据**，分析近期的价格波动、成交量变化和关键支撑/阻力位。
        - 结合**长期数据**，分析K线形态、移动平均线（MA5, MA10, MA20）的交叉情况（金叉/死叉），判断当前股票的整体趋势（上涨、下跌、震荡）。
    3.  **综合投资建议**：综合新闻基本面和技术面分析，给出明确的、可操作的投资建议（如：建议买入、持有、卖出、或观望），并说明理由和潜在风险。

    **输入信息：**
    - **新闻内容**：{news}
    - **短期（7日）数据**：
    {stock_data_short_str}
    - **长期（30日）数据摘要**：
    {stock_data_long_str}

    **请严格按照以下格式输出你的分析报告：**
    ---
    ### 核心新闻解读
    [在此处填写你对新闻的分析]

    ### 技术面走势分析
    **短期（周线级别）来看**，[在此处填写你对7日数据的分析]。
    **长期（月线级别）来看**，[在此处填写你对30日数据的分析，特别是K线和均线趋势]。

    ### 综合投资建议
    **结论**：[在此处填写“建议买入/持有/卖出/观望”]
    **理由**：[在此处填写你的核心理由]。
    **风险提示**：[在此处提示潜在的风险]。
    ---
    """
    return gemini_prompt(prompt)
