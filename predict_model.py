import torch
from transformers import BertTokenizer
# 从 model.py 导入模型结构和切块函数
from model import DualInputFakeNewsClassifier, split_text_into_chunks
# === 新增：导入我们的情感分析器 ===
from sentiment_analyzer import get_article_sentiment

HF_MODEL_URL = "https://huggingface.co/YanRY/Chinese-financial-news/resolve/main/best_fakenews_modelv3.pt"
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global _model
    if _model is None:
        _model = DualInputFakeNewsClassifier()
        state_dict = torch.hub.load_state_dict_from_url(HF_MODEL_URL, map_location=_device, progress=True)
        _model.load_state_dict(state_dict)
        _model.eval().to(_device)
    return _model

def clean_text_for_bert(text):
    # 这是给BERT的简单清洗，可以和情感分析的清洗分开
    return ' '.join(text.strip().split())

def predict_by_model(title, content, platform_code):
    model = load_model()
    
    # === 关键步骤 1: 计算情感得分 ===
    # 调用我们新创建的模块来获取整篇文章的情感分
    sentiment_score = get_article_sentiment(title, content)
    print(f"计算出的情感分: {sentiment_score}") # 方便调试

    # --- 后续流程与之前类似，但使用计算出的情感分 ---
    
    full_text = clean_text_for_bert(title + "。" + content)

    # 1. 文本切块 (Chunking)
    max_chunks = 4
    max_length = 512
    chunks = split_text_into_chunks(full_text, tokenizer, max_tokens=max_length, max_chunks=max_chunks)

    if not chunks:
        return "无法判断", [0.5, 0.5]

    # 2. 对每个块进行编码
    input_ids_list = []
    attention_mask_list = []
    for chunk in chunks:
        encoded = tokenizer(
            chunk,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])

    # 3. 补齐到 max_chunks
    while len(input_ids_list) < max_chunks:
        input_ids_list.append(torch.zeros((1, max_length), dtype=torch.long))
        attention_mask_list.append(torch.zeros((1, max_length), dtype=torch.long))

    # 4. 堆叠成 batch
    input_ids = torch.cat(input_ids_list, dim=0).unsqueeze(0)
    attention_mask = torch.cat(attention_mask_list, dim=0).unsqueeze(0)

    # === 关键步骤 2: 使用计算出的情感分准备上下文特征 ===
    # context_features: [发布来源, 文本情绪]
    context_feat = torch.tensor([[platform_code, sentiment_score]])

    # 6. 移动到设备并进行预测
    inputs = {
        'input_ids': input_ids.to(_device),
        'attention_mask': attention_mask.to(_device),
        'context_features': context_feat.to(_device).float()
    }

    with torch.no_grad():
        logits = model(**inputs)
        prob = torch.softmax(logits, dim=1)[0]
        
    result_label = "真实" if prob[1] > 0.5 else "虚假"
    return result_label, prob.cpu().tolist()
