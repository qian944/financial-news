import pandas as pd
import jieba
import re
import os

# --- 全局变量，只加载一次资源，提高效率 ---
_sentiment_resources = {}

def _load_resources():
    """加载情感词典和停用词，并缓存起来。"""
    global _sentiment_resources
    if _sentiment_resources:
        return _sentiment_resources

    print("首次加载情感分析资源...")
    
    # 词典和停用词路径 (假设在项目根目录)
    dict_path = "中文金融情感词典_姜富伟等(2020).xlsx"
    stopwords_path = "cn_stopwords.txt"

    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"情感词典文件未找到: {dict_path}")
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"停用词文件未找到: {stopwords_path}")

    # 读取正负词表
    pos_words = pd.read_excel(dict_path, sheet_name="positive")['Positive Word']
    neg_words = pd.read_excel(dict_path, sheet_name="negative")['Negative Word']
    
    # 清洗空值和空白字符
    positive_set = set(w.strip() for w in pos_words.dropna() if isinstance(w, str))
    negative_set = set(w.strip() for w in neg_words.dropna() if isinstance(w, str))

    # 读取停用词表
    with open(stopwords_path, encoding="utf-8") as f:
        stopwords_set = set(line.strip() for line in f if line.strip())

    _sentiment_resources = {
        "positive_set": positive_set,
        "negative_set": negative_set,
        "stopwords_set": stopwords_set
    }
    print("情感分析资源加载完成。")
    return _sentiment_resources

def clean_chinese_text_for_sentiment(text):
    """专门用于情感分析的文本清洗（分词+去停用词）。"""
    if not isinstance(text, str):
        return ""
    # 保留中文、数字、英文字母（去掉标点、特殊字符）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    # 分词
    words = jieba.lcut(text)
    # 去除停用词 + 空白词
    resources = _load_resources()
    cleaned = [word for word in words if word not in resources["stopwords_set"] and word.strip()]
    return ' '.join(cleaned)

def compute_sentiment_score(text):
    """计算单个文本的情感得分。"""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    
    resources = _load_resources()
    pos_set = resources["positive_set"]
    neg_set = resources["negative_set"]
    
    words = jieba.lcut(text)
    if not words:
        return 0.0

    pos_count = sum(1 for w in words if w in pos_set)
    neg_count = sum(1 for w in words if w in neg_set)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0

    score = (pos_count - neg_count) / (abs(pos_count) + abs(neg_count))
    return round(score, 4)

def get_article_sentiment(title, content):
    """
    计算整篇文章（标题+正文）的加权情感分。
    这是最终提供给外部调用的主函数。
    """
    # 清洗用于分词和情感计算
    title_cleaned = clean_chinese_text_for_sentiment(title)
    content_cleaned = clean_chinese_text_for_sentiment(content)

    # 计算各自的情感分
    title_sentiment = compute_sentiment_score(title_cleaned)
    content_sentiment = compute_sentiment_score(content_cleaned)
    
    # 按照训练时的逻辑进行加权
    # 文章情绪词典分 = (0.6 * 标题情绪词典分 + 0.4 * 正文情绪词典分)
    final_sentiment = (title_sentiment + content_sentiment)
    
    return round(final_sentiment, 4)
