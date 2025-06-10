import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 定义 Hugging Face 模型权重 URL（你应替换为实际链接）
HF_MODEL_URL = "https://huggingface.co/your_username/your_model_name/resolve/main/news_model.pth"

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型结构（需与训练时一致）
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

    # 下载并加载权重
    state_dict = torch.hub.load_state_dict_from_url(HF_MODEL_URL, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, device

# 初始化一次（避免每次预测都重复下载）
model, device = load_model()
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def clean_text(text):
    return ' '.join(text.split())

def predict_by_model(title, content, platform_code):
    title = clean_text(title)
    content = clean_text(content)
    combined = f"{title} {content} 平台类型：{platform_code}"

    inputs = tokenizer(combined, return_tensors='pt', truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.softmax(outputs.logits, dim=1)[0]

    return '真实' if prob[1] > 0.5 else '虚假', prob.tolist()
