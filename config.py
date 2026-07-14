"""
配置文件 - 修改API Key等信息在此文件
"""
import os
from pathlib import Path

# ====== 大模型配置（硅基流动 SiliconFlow 免费API）======
# 获取免费API Key: https://cloud.siliconflow.cn/account/ak
# 注册即送额度，使用Qwen2-7B模型完全免费
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-wxgdvbkmoitqvzmqgoedqccudgljgirkenvllfuzgcpfkibi")

# 接口地址（无需修改）
API_BASE_URL = "https://api.siliconflow.cn/v1"

# 模型名称 - 使用免费的Qwen2.5-7B-Instruct
# 其他可选免费模型: Qwen/Qwen3-8B-Instruct, GLM-4-Flash
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 模型参数
MODEL_TEMPERATURE = 0.1       # 较低温度，确保回答准确不编造
MODEL_MAX_TOKENS = 2048       # 最大输出长度
MODEL_TOP_P = 0.9

# ====== 向量库配置 ======
# Chroma 持久化存储路径
CHROMA_PERSIST_DIR = str(Path(__file__).parent / "chroma_db")

# 文档分块参数
CHUNK_SIZE = 300              # 每块最大字符数
CHUNK_OVERLAP = 50            # 块之间重叠字符数

# ====== Embedding 模型配置 ======
# 使用硅基流动的 embedding 模型（免费）
EMBEDDING_MODEL = "BAAI/bge-m3"
# OpenAIEmbeddings 会自动追加 /embeddings 路径，这里只传基础地址
EMBEDDING_API_URL = API_BASE_URL

# ====== 村名配置 ======
VILLAGE_NAME = "李营村"
VILLAGE_FULL = "河南省南阳市镇平县石佛寺镇李营村"

# ====== 系统提示词 ======
SYSTEM_PROMPT = """你是一个村级便民政策AI问答助手，服务于李营村。你的职责是：
1. 只基于提供的政策文档内容回答村民的问题
2. 如果文档中没有相关信息，请回答："抱歉，本村当前政策文件中没有找到相关信息，请咨询村委会工作人员。"
3. 不要编造或推测任何政策内容
4. 村民可能用大白话提问，请用通俗易懂的语言回答
5. 回答要简洁清晰，关键信息（金额、日期、条件等）用**加粗**强调
6. 必须使用简体中文回复，不得使用繁体中文
7. 如果村民问的问题与政策无关，请回复："抱歉，我只能解答本村政策相关问题哦～"
8. 回答末尾请注明："（以上信息仅供参考，具体政策以村委会解释为准）"

参考政策内容：
{context}

村民问题：{question}
"""

# ====== 村内地点导览 ======
# 坐标可在此拾取：https://lbs.amap.com/console/show/picker
VILLAGE_LOCATIONS = [
    {
        "id": "cunweihui",
        "name": "李营村委会",
        "desc": "村委办公地点，办理盖章、证明、申请等各类村级事务",
        "lat": 33.0520,
        "lng": 112.0830,
        "icon": "🏛️",
    },
    {
        "id": "weishengsuo",
        "name": "村卫生所",
        "desc": "医保定点门诊，常见病诊疗、慢病取药",
        "lat": 33.0515,
        "lng": 112.0825,
        "icon": "🏥",
    },
    {
        "id": "laonian",
        "name": "老年活动中心",
        "desc": "老年人休闲娱乐、养老政策咨询",
        "lat": 33.0523,
        "lng": 112.0832,
        "icon": "☕",
    },
    {
        "id": "guangchang",
        "name": "文化广场",
        "desc": "村内活动集会场地，政策宣传公示栏所在地",
        "lat": 33.0518,
        "lng": 112.0828,
        "icon": "🏟️",
    },
    {
        "id": "xiaomai",
        "name": "便民超市",
        "desc": "日常生活用品购买，社保代缴点",
        "lat": 33.0512,
        "lng": 112.0835,
        "icon": "🏪",
    },
]

# ====== 安全过滤关键词 ======
# 当问题包含以下关键词时，直接拒绝回答（不触发RAG）
NON_POLICY_KEYWORDS = [
    "你好", "你是谁", "你能干什么", "你能做什么",
    "天气", "新闻", "股票", "游戏", "电影",
    "政治", "色情", "赌博", "毒品",
]

# ====== 预设快捷问题 ======
QUICK_QUESTIONS = [
    "医保怎么报销",
    "养老金怎么领",
    "种地补贴怎么申请",
    "低保怎么办理",
    "宅基地怎么申请",
]
