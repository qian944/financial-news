import os
import torch
from transformers import BertTokenizer
from model import DualInputFakeNewsClassifier

HF_MODEL_URL = "https://huggingface.co/your_user/your_repo/resolve/main/model_final.pt"
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

def clean_text(text):
    return ' '.join(text.strip().split())

def predict_by_model(title, content, platform_code):
    model = load_model()
    text = clean_text(title + " " + content)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    context_feat = torch.tensor([[platform_code, 0.0]])  # 可添加情绪分析
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    context_feat = context_feat.to(_device).float()
    with torch.no_grad():
        logits = model(**inputs, context_feats=context_feat)
        prob = torch.softmax(logits, dim=1)[0]
    return "真实" if prob[1] > 0.5 else "虚假", prob.tolist()
