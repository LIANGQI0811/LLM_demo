import paddle
import paddle.nn as nn
import math

# ===================== 1. 基础配置 =====================
# 设备配置
DEVICE = paddle.get_device()
print(f"Using device: {DEVICE}")

# 词表配置
VOCAB = {
    "<PAD>": 0,  # 填充token
    "<UNK>": 1,  # 未知token
    "我": 2,
    "喜欢": 3,
    "吃": 4,
    "苹果": 5,
    "用": 6,
    "手机": 7,
    "DeepSeek": 8,
    "百度": 9,
    "搜索": 10
}
VOCAB_SIZE = len(VOCAB)          # 词表大小
EMBEDDING_DIM = 128             # 嵌入维度
MAX_SEQ_LEN = 32                # 最大序列长度
N_LAYERS = 2                    # Transformer层数
N_HEADS = 4                     # 注意力头数
FFN_DIM = 256                   # 前馈神经网络隐藏层维度
DROPOUT = 0.1                   #  dropout概率

# ===================== 2. Token化函数 =====================
def tokenize(text, vocab, max_seq_len):
    """
    文本→Token索引序列
    :param text: 输入文本
    :param vocab: 词表
    :param max_seq_len: 最大序列长度
    :return: token索引张量 (1, max_seq_len)
    """
    # 模拟分词
    tokens = text.split()
    # 转索引
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # 截断/填充
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
    else:
        token_ids += [vocab["<PAD>"]] * (max_seq_len - len(token_ids))
    # 转为Paddle张量
    return paddle.to_tensor([token_ids], dtype="int64")

# ===================== 3. 词嵌入层（含位置编码） =====================
class TransformerEmbedding(nn.Layer):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        # 词嵌入层
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        # 位置编码
        self.position_embedding = self._build_sinusoidal_position_embedding(max_seq_len, embed_dim)
        # 归一化
        self.norm = nn.LayerNorm(embed_dim)
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化词嵌入矩阵"""
        # 初始化所有词嵌入
        nn.initializer.Normal(mean=0.0, std=0.02)(self.token_embedding.weight)
        # 使用with语句临时设置PAD的嵌入为0
        with paddle.no_grad():
            # 获取权重的numpy数组
            weight_np = self.token_embedding.weight.numpy()
            # 将第一个元素（PAD）设置为0
            weight_np[0] = 0.0
            # 将修改后的值设置回权重
            self.token_embedding.weight.set_value(paddle.to_tensor(weight_np))

    def _build_sinusoidal_position_embedding(self, max_seq_len, embed_dim):
        """构建正弦位置编码"""
        position = paddle.arange(0, max_seq_len, dtype="float32").unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_emb = paddle.zeros([max_seq_len, embed_dim])
        pos_emb[:, 0::2] = paddle.sin(position * div_term)
        pos_emb[:, 1::2] = paddle.cos(position * div_term)
        return pos_emb.unsqueeze(0)

    def forward(self, token_ids):
        """
        前向传播：词嵌入 + 位置编码 → 归一化
        :param token_ids: token索引张量，shape [batch_size, seq_len]
        :return: 最终嵌入向量，shape [batch_size, seq_len, embed_dim]
        """
        # 词嵌入
        token_emb = self.token_embedding(token_ids)
        # 词嵌入 + 位置编码
        seq_len = token_ids.shape[1]
        embed = token_emb + self.position_embedding[:, :seq_len, :]
        # 归一化
        embed = self.norm(embed)
        return embed

# ===================== 4. 多头注意力机制 =====================
class MultiHeadAttention(nn.Layer):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.initializer.Normal(mean=0.0, std=0.02)(self.q_proj.weight)
        nn.initializer.Normal(mean=0.0, std=0.02)(self.k_proj.weight)
        nn.initializer.Normal(mean=0.0, std=0.02)(self.v_proj.weight)
        nn.initializer.Normal(mean=0.0, std=0.02)(self.out_proj.weight)

    def forward(self, q, k, v, mask=None):
        """
        前向传播
        :param q: 查询张量，shape [batch_size, seq_len_q, embed_dim]
        :param k: 键张量，shape [batch_size, seq_len_k, embed_dim]
        :param v: 值张量，shape [batch_size, seq_len_v, embed_dim]
        :param mask: 掩码张量，shape [batch_size, seq_len_q, seq_len_k]
        :return: 注意力输出，shape [batch_size, seq_len_q, embed_dim]
        """
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_k = k.shape[1]
        
        # 线性变换
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 重塑为多头
        q = q.reshape([batch_size, seq_len_q, self.n_heads, self.head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([batch_size, seq_len_k, self.n_heads, self.head_dim]).transpose([0, 2, 3, 1])
        v = v.reshape([batch_size, seq_len_k, self.n_heads, self.head_dim]).transpose([0, 2, 1, 3])
        
        # 计算注意力分数
        attn_scores = paddle.matmul(q, k) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            attn_scores = paddle.where(mask == 0, paddle.to_tensor(-1e9), attn_scores)
        
        # 计算注意力权重
        attn_weights = nn.functional.softmax(attn_scores, axis=-1)
        
        # 加权求和
        attn_output = paddle.matmul(attn_weights, v)
        
        # 重塑为原始形状
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, seq_len_q, self.embed_dim])
        
        # 输出线性变换
        output = self.out_proj(attn_output)
        
        return output

# ===================== 5. 前馈神经网络 =====================
class FeedForward(nn.Layer):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.initializer.Normal(mean=0.0, std=0.02)(self.linear1.weight)
        nn.initializer.Normal(mean=0.0, std=0.02)(self.linear2.weight)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，shape [batch_size, seq_len, embed_dim]
        :return: 输出张量，shape [batch_size, seq_len, embed_dim]
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ===================== 6. Transformer解码器层 =====================
class TransformerDecoderLayer(nn.Layer):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = MultiHeadAttention(embed_dim, n_heads)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # 前馈神经网络层
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入张量，shape [batch_size, seq_len, embed_dim]
        :param mask: 掩码张量，shape [batch_size, seq_len, seq_len]
        :return: 输出张量，shape [batch_size, seq_len, embed_dim]
        """
        # 自注意力层
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈神经网络层
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

# ===================== 7. 完整的LLM模型 =====================
class PaddleLLM(nn.Layer):
    def __init__(self, vocab_size, embed_dim, max_seq_len, n_layers, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # 词嵌入层
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, max_seq_len)
        
        # Transformer解码器层
        self.decoder_layers = nn.LayerList([
            TransformerDecoderLayer(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.initializer.Normal(mean=0.0, std=0.02)(self.output_layer.weight)

    def forward(self, input_ids):
        """
        前向传播
        :param input_ids: 输入token索引，shape [batch_size, seq_len]
        :return: 输出logits，shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # 生成掩码
        mask = paddle.tril(paddle.ones([seq_len, seq_len])).unsqueeze(0).expand([batch_size, seq_len, seq_len])
        
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 遍历解码器层
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        
        # 输出层
        logits = self.output_layer(x)
        
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=5):
        """
        生成文本
        :param input_ids: 输入token索引，shape [1, seq_len]
        :param max_length: 生成的最大长度
        :param temperature: 温度参数
        :param top_k: Top-K采样参数
        :return: 生成的token索引，shape [1, max_length]
        """
        self.eval()
        
        for i in range(max_length - input_ids.shape[1]):
            # 前向传播
            logits = self(input_ids)
            
            # 获取最后一个token的logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-K采样
            top_k_values, top_k_indices = paddle.topk(next_token_logits, k=top_k)
            
            # 计算概率
            probabilities = nn.functional.softmax(top_k_values, axis=-1)
            
            # 采样
            next_token = paddle.multinomial(probabilities, num_samples=1)
            next_token = paddle.gather(top_k_indices, axis=-1, index=next_token)
            
            # 拼接
            input_ids = paddle.concat([input_ids, next_token], axis=1)
            
        return input_ids

# ===================== 8. 测试运行 =====================
if __name__ == "__main__":
    # 初始化模型
    model = PaddleLLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQ_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT
    )
    print("Paddle LLM模型初始化完成！")
    
    # 输入文本
    input_text = "我 喜欢"
    print(f"\n输入文本：{input_text}")
    
    # Token化
    input_ids = tokenize(input_text, VOCAB, MAX_SEQ_LEN)
    print(f"Token索引：{input_ids.numpy()}")
    print(f"Token索引形状：{input_ids.shape}")
    
    # 前向传播
    logits = model(input_ids)
    print(f"\n输出logits形状：{logits.shape}")
    
    # 生成文本
    generated_ids = model.generate(input_ids, max_length=10)
    print(f"\n生成的Token索引：{generated_ids.numpy()}")
    
    # 将Token索引转换为文本
    id_to_token = {v: k for k, v in VOCAB.items()}
    generated_text = " ".join([id_to_token[id.item()] for id in generated_ids[0]])
    print(f"生成的文本：{generated_text}")
