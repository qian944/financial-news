import torch
from transformers import BertTokenizer
# 从 model.py 导入我们刚刚修改好的类和函数
from model import DualInputFakeNewsClassifier, split_text_into_chunks

# HF_MODEL_URL = "https://huggingface.co/YanRY/Chinese-financial-news/resolve/main/best_fakenews_modelv2.pt" # v2
HF_MODEL_URL = "https://huggingface.co/YanRY/Chinese-financial-news/resolve/main/best_fakenews_modelv3.pt" # 您的代码用的是v3，这里保持一致
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load_model 函数现在不需要改动，因为它加载的是我们定义好的正确模型
def load_model():
    global _model
    if _model is None:
        # 初始化正确的模型结构
        _model = DualInputFakeNewsClassifier()
        # 加载权重
        state_dict = torch.hub.load_state_dict_from_url(HF_MODEL_URL, map_location=_device, progress=True)
        _model.load_state_dict(state_dict)
        _model.eval().to(_device)
    return _model

def clean_text(text):
    return ' '.join(text.strip().split())

# === 关键步骤 4: 重写预测函数以匹配新的数据处理流程 ===
def predict_by_model(title, content, platform_code):
    model = load_model()
    full_text = clean_text(title + "。" + content)

    # 1. 文本切块 (Chunking)
    max_chunks = 4
    max_length = 512
    chunks = split_text_into_chunks(full_text, tokenizer, max_tokens=max_length, max_chunks=max_chunks)

    # 如果没有切出任何内容（比如输入为空），返回默认值
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
    input_ids = torch.cat(input_ids_list, dim=0).unsqueeze(0) # 增加 batch 维度 -> [1, 4, 512]
    attention_mask = torch.cat(attention_mask_list, dim=0).unsqueeze(0) # -> [1, 4, 512]

    # 5. 准备上下文特征
    # 训练时用了2个特征：情绪和来源。这里我们暂时只用来源，情绪设为0
    context_feat = torch.tensor([[platform_code, 0.0]])  # 假设platform_code是数值

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
