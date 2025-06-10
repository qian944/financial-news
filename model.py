import torch
import torch.nn as nn
from transformers import BertModel
import re

# === 关键步骤 1: 复制训练时的 Attention 类 ===
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1) # 训练代码里的 attention 貌似没有 bias，这里保持一致

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_dim]
        scores = self.attention(lstm_out).squeeze(-1)      # [batch, seq_len]
        weights = torch.softmax(scores, dim=1)             # [batch, seq_len]
        weighted_sum = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # [batch, hidden_dim]
        return weighted_sum

# === 关键步骤 2: 复制训练时的模型定义，并稍作修改 ===
class DualInputFakeNewsClassifier(nn.Module):
    # 修改__init__，不再需要 bert_model_path 参数，直接使用预设值
    def __init__(self, context_input_dim=2, context_hidden_dim=16, final_hidden_dim=128, num_labels=2):
        super().__init__()

        # 直接加载预训练模型
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bert_hidden_dim = self.bert.config.hidden_size

        # 使用和训练时完全一致的层定义
        self.bilstm = nn.LSTM(
            input_size=self.bert_hidden_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(hidden_dim=512)  # BiLSTM输出维度为 256*2 = 512

        # 上下文特征 MLP
        self.context_net = nn.Sequential(
            nn.Linear(context_input_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 最终拼接 + 分类层
        fusion_input_dim = 512 + context_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_hidden_dim, num_labels)
        )

    # forward 方法保持和训练时一致的逻辑
    def forward(self, input_ids, attention_mask, context_features):
        """
        input_ids: [batch, num_chunks, 512]
        attention_mask: [batch, num_chunks, 512]
        context_features: [batch, 2]
        """
        batch_size, num_chunks, _ = input_ids.size()
        cls_list = []

        for i in range(num_chunks):
            input_ids_i = input_ids[:, i, :]           # [batch, 512]
            attn_mask_i = attention_mask[:, i, :]      # [batch, 512]

            bert_out = self.bert(input_ids=input_ids_i, attention_mask=attn_mask_i, return_dict=True)
            cls_i = bert_out.last_hidden_state[:, 0, :]   # 取 [CLS] 向量：[batch, 768]
            cls_list.append(cls_i)

        cls_seq = torch.stack(cls_list, dim=1) # [batch, num_chunks, 768]

        lstm_out, _ = self.bilstm(cls_seq)             # [batch, num_chunks, 512]
        news_vector = self.attention(lstm_out)         # [batch, 512]

        context_vector = self.context_net(context_features)

        fusion = torch.cat([news_vector, context_vector], dim=1)

        logits = self.classifier(fusion)
        return logits


# === 关键步骤 3: 复制文本切分函数 ===
def split_text_into_chunks(text, tokenizer, max_tokens=512, max_chunks=4):
    if not isinstance(text, str):
        return []

    sentences = re.split(r'(?<=[。！？!？])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    all_chunks = []
    current_chunk = ""

    for sent in sentences:
        # 检查添加新句子后是否会超长
        temp_chunk = current_chunk + sent
        if len(tokenizer.tokenize(temp_chunk)) > max_tokens - 2: # -2 for [CLS] and [SEP]
            if current_chunk:
                all_chunks.append(current_chunk)
            current_chunk = sent
        else:
            current_chunk = temp_chunk
    
    if current_chunk:
        all_chunks.append(current_chunk)

    if len(all_chunks) > max_chunks:
        return all_chunks[:2] + all_chunks[-2:]
    else:
        return all_chunks
