import torch
import torch.nn as nn
import math

# ===================== 1. 基础配置（贴合DeepSeek 6B简化版） =====================
# 设备配置：优先GPU（大模型必备）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 词表：模拟中文优化的词表（DeepSeek真实词表约10万，这里简化）
VOCAB = {
    "<PAD>": 0,  # 填充token
    "<UNK>": 1,  # 未知token
    "我": 2,
    "喜欢": 3,
    "吃": 4,
    "苹果": 5,
    "用": 6,
    "手机": 7,
    "DeepSeek": 8
}
VOCAB_SIZE = len(VOCAB)          # 词表大小
EMBEDDING_DIM = 128             #   （DeepSeek 6B真实值为4096，这里简化）
MAX_SEQ_LEN = 32                # 最大序列长度（位置编码需要）

# ===================== 2. Token化函数（模拟大模型分词） =====================
def tokenize(text, vocab, max_seq_len):
    """
    文本→Token索引序列（模拟大模型BPE分词）
    :param text: 输入文本
    :param vocab: 词表
    :param max_seq_len: 最大序列长度，超长截断、不足补PAD
    :return: token索引张量 (1, max_seq_len)
    """
    # 模拟分词（真实大模型用BPE，这里简化为按空格/固定词拆分）
    tokens = text.split()  # 示例："我 喜欢 吃 苹果" → ["我", "喜欢", "吃", "苹果"]
    # 转索引（未知词用<UNK>）
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # 截断/填充
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
    else:
        token_ids += [vocab["<PAD>"]] * (max_seq_len - len(token_ids))
    # 转为张量并放到指定设备
    return torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

# ===================== 3. 词嵌入层（含位置编码，大模型核心） =====================
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, device):
        super().__init__()
        # 核心1：词嵌入层（可训练参数，大模型中这是输入层第一步）
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,    # 词表大小
            embedding_dim=embed_dim,      # 嵌入维度
            padding_idx=0,                # <PAD>的索引，计算损失时会忽略
            device=device
        )
        # 核心2：位置编码（正弦编码，DeepSeek/GPT均支持，RoPE可替换此处）
        self.position_embedding = self._build_sinusoidal_position_embedding(max_seq_len, embed_dim, device)
        # 归一化（大模型Pre-Norm架构必备）
        self.norm = nn.LayerNorm(embed_dim, device=device)
        # 嵌入层参数初始化（Transformer论文标准）
        self._init_weights()

    def _init_weights(self):
        """初始化词嵌入矩阵（均值0，标准差0.02，大模型标配）"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # <PAD>的嵌入向量初始化为0
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0.0)

    def _build_sinusoidal_position_embedding(self, max_seq_len, embed_dim, device):
        """构建正弦位置编码（替代RoPE，易理解）"""
        # 位置索引：0,1,2,...,max_seq_len-1
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 频率因子：1/(10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        # 初始化位置编码矩阵
        pos_emb = torch.zeros(max_seq_len, embed_dim, device=device)
        # 偶数位用sin，奇数位用cos
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        # 扩展维度：[max_seq_len, embed_dim] → [1, max_seq_len, embed_dim]
        return pos_emb.unsqueeze(0)

    def forward(self, token_ids):
        """
        前向传播：词嵌入 + 位置编码 → 归一化
        :param token_ids: token索引张量，shape [batch_size, seq_len]
        :return: 最终嵌入向量，shape [batch_size, seq_len, embed_dim]
        """
        # 1. 词嵌入：[batch_size, seq_len] → [batch_size, seq_len, embed_dim]
        token_emb = self.token_embedding(token_ids)
        # 2. 词嵌入 + 位置编码（广播相加）
        embed = token_emb + self.position_embedding[:, :token_ids.size(1), :]
        # 3. 归一化（Pre-Norm，大模型标准操作）
        embed = self.norm(embed)
        return embed

# ===================== 4. 测试运行（完整流程） =====================
if __name__ == "__main__":
    # 步骤1：初始化嵌入层（模拟DeepSeek 6B嵌入层）
    embedding_layer = TransformerEmbedding(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE
    )
    print("嵌入层初始化完成！")
    
    # 步骤2：输入文本并token化
    input_text = "我 喜欢 吃 苹果"  # 注意空格分隔，模拟分词结果
    token_ids = tokenize(input_text, VOCAB, MAX_SEQ_LEN)
    print(f"\n输入文本：{input_text}")
    print(f"Token索引：{token_ids}")
    print(f"Token索引形状：{token_ids.shape}")  # [1, 32]（batch_size=1，seq_len=32）
    
    # 步骤3：前向传播获取嵌入向量
    final_embeddings = embedding_layer(token_ids)
    print(f"\n最终嵌入向量形状：{final_embeddings.shape}")  # [1, 32, 128]
    print(f"第一个token（我）的嵌入向量前5个值：{final_embeddings[0][0][:5]}")
