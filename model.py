import torch.nn as nn
from transformers import BertModel

class DualInputFakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bi_lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 2, 64),  # 2 是上下文特征：平台类型 & 情绪
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, context_feats):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.bi_lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        weighted_output = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        combined = torch.cat([weighted_output, context_feats], dim=1)
        return self.classifier(combined)
