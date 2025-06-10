import torch
from transformers import BertTokenizer, BertForSequenceClassification
from hf_utils import download_model_if_needed

# 下载模型权重
download_model_if_needed()

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
model.load_state_dict(torch.load("downloaded/news_model.pt", map_location=torch.device('cpu')))
model.eval()

def clean_text(text):
    return ' '.join(text.split())

def predict_by_model(title, content, platform_code):
    title = clean_text(title)
    content = clean_text(content)
    combined = f"{title} {content} 平台类型：{platform_code}"
    inputs = tokenizer(combined, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.softmax(outputs.logits, dim=1)[0]
    return '真实' if prob[1] > 0.5 else '虚假', prob.tolist()
