"""
李营村便民政策AI问答助手 - Streamlit 网页版
基于RAG技术，使用硅基流动免费API提供政策问答服务
"""
import os
import glob
import logging
from pathlib import Path
from threading import Lock

import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI, AuthenticationError, BadRequestError, RateLimitError, APITimeoutError

# ====== API Key 解析（优先级：st.secrets > 环境变量 > config.py默认值）======
if "SILICONFLOW_API_KEY" in st.secrets:
    os.environ["SILICONFLOW_API_KEY"] = st.secrets["SILICONFLOW_API_KEY"]

from config import *

# ====== 页面配置 ======
st.set_page_config(
    page_title=f"{VILLAGE_NAME}便民政策AI助手",
    page_icon="🏘️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ====== 路径配置 ======
BASE_DIR = Path(__file__).parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
POLICIES_DIR = BASE_DIR / "policies"

# ====== 日志 ======
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    encoding="utf-8", force=True)
logger = logging.getLogger(__name__)

# ====== 全局锁 ======
vector_lock = Lock()


# ====================================================================
#  缓存资源（Streamlit 缓存，避免每次交互都重新加载）
# ====================================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """创建Embedding模型实例"""
    return OpenAIEmbeddings(
        openai_api_key=API_KEY,
        openai_api_base=EMBEDDING_API_URL,
        model=EMBEDDING_MODEL,
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """加载已有向量库，若不存在则从政策文件构建"""
    embeddings = get_embeddings()

    # 尝试加载已有向量库
    if os.path.exists(os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3")):
        try:
            vs = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
            logger.info("向量库加载成功")
            return vs
        except Exception as e:
            logger.warning(f"加载已有向量库失败，将重建: {e}")

    # 不存在或加载失败，从政策文件构建
    docs = load_policy_documents()
    chunks = split_documents(docs)
    if not chunks:
        logger.warning("没有加载到任何政策文档！")
        return None

    try:
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info(f"向量库构建完成，共 {len(chunks)} 个文本块")
        return vs
    except AuthenticationError:
        logger.error("API Key认证失败，无法构建向量库")
        return None
    except Exception as e:
        logger.error(f"构建向量库失败: {e}")
        return None


def rebuild_vectorstore():
    """清空缓存，下次访问时重建向量库"""
    st.cache_resource.clear()


# ====================================================================
#  文档处理函数（从 app.py 移植）
# ====================================================================

def load_policy_documents():
    """从政策目录加载所有 .txt 和 .md 文件"""
    docs = []
    supported_exts = ["*.txt", "*.md"]
    for ext in supported_exts:
        pattern = str(POLICIES_DIR / "**" / ext)
        for filepath in glob.glob(pattern, recursive=True):
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                docs.extend(loader.load())
                logger.info(f"已加载政策文件: {filepath}")
            except Exception as e:
                logger.warning(f"加载文件失败 {filepath}: {e}")
    return docs


def split_documents(docs):
    """将文档切分为小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"文档切分为 {len(chunks)} 个文本块")
    return chunks


# ====================================================================
#  安全过滤
# ====================================================================

def is_policy_related(question):
    """关键词匹配判断是否与政策相关"""
    policy_keywords = [
        "医保", "养老", "补贴", "低保", "宅基地", "报销",
        "缴费", "保险", "耕地", "农机", "贷款", "补助",
        "村委会", "办理", "申请", "政策", "怎么", "如何",
        "多少", "什么", "哪里", "条件", "标准",
    ]
    for kw in policy_keywords:
        if kw in question.strip():
            return True
    return False


def check_non_policy(question):
    """检查是否触发了非政策关键词"""
    q = question.strip().lower()
    for kw in NON_POLICY_KEYWORDS:
        if kw in q:
            return True
    return False


# ====================================================================
#  大模型调用
# ====================================================================

def call_llm(system_prompt, user_message):
    """调用硅基流动 API"""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=MODEL_TEMPERATURE,
            max_tokens=MODEL_MAX_TOKENS,
            top_p=MODEL_TOP_P,
        )
        return response.choices[0].message.content
    except AuthenticationError:
        return "⚠️ API Key 认证失败，请检查 API_KEY 是否正确配置。"
    except (BadRequestError, RateLimitError) as e:
        msg = str(e).lower()
        if "insufficient" in msg or "quota" in msg or "balance" in msg:
            return "⚠️ 账户额度不足，请到硅基流动平台充值或更换API Key。"
        return f"⚠️ API请求异常：{str(e)[:100]}"
    except APITimeoutError:
        return "⚠️ 请求超时，请检查网络连接后重试。"
    except Exception as e:
        logger.error(f"调用大模型失败: {e}")
        return f"⚠️ 服务暂时不可用，请稍后重试。"


# ====================================================================
#  核心问答处理
# ====================================================================

def process_question(question):
    """RAG检索 + LLM生成回答"""
    # 安全检查
    if check_non_policy(question):
        return "抱歉，我只能解答本村政策相关问题哦～", []

    if not is_policy_related(question):
        return "抱歉，我只能解答本村政策相关问题哦～", []

    # 向量检索
    vs = get_vectorstore()
    if vs is None:
        return "⚠️ 政策知识库尚未加载，请先检查 policies/ 目录中是否有政策文件。", []

    try:
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        docs = retriever.invoke(question)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        docs = []

    if not docs:
        return "抱歉，本村当前政策文件中没有找到相关信息，请咨询村委会工作人员。", []

    # 构建上下文和来源
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([
        os.path.basename(d.metadata.get("source", "未知文件"))
        for d in docs if d.metadata.get("source")
    ]))

    # 构建对话历史上下文
    history_text = get_history_text()
    if history_text:
        user_message = f"对话历史：\n{history_text}\n\n最新问题：{question}"
    else:
        user_message = question

    # 调用大模型
    system_prompt = SYSTEM_PROMPT.format(村名=VILLAGE_NAME, context=context, question="{question}")
    answer = call_llm(system_prompt, user_message)
    answer = format_answer(answer)

    return answer, sources


def get_history_text(max_pairs=3):
    """从会话状态中提取最近对话历史文本"""
    history_msgs = []
    count = 0
    for msg in reversed(st.session_state.messages):
        if count >= max_pairs:
            break
        if msg["role"] in ("user", "assistant"):
            label = "村民" if msg["role"] == "user" else "助手"
            history_msgs.insert(0, f"{label}: {msg['content']}")
            if msg["role"] == "assistant":
                count += 1
    return "\n".join(history_msgs)


def format_answer(text):
    """格式化回答，确保段落清晰"""
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ====================================================================
#  会话状态初始化
# ====================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"您好！我是 **{VILLAGE_NAME}便民政策AI助手**，您可以问我关于 "
                "**医保、养老、种地补贴、低保、宅基地** 等政策问题，"
                "我会根据咱村的政策文件为您解答！"
            ),
        }
    ]

if "show_location_status" not in st.session_state:
    st.session_state.show_location_status = False


# ====================================================================
#  UI: 顶部标题
# ====================================================================

st.markdown(f"""
<div style="text-align:center;padding:16px 16px 12px;
     background:linear-gradient(135deg,#1a73e8,#1557b0);color:white;
     border-radius:12px;margin-bottom:16px;">
  <h1 style="margin:0;font-size:28px;letter-spacing:2px;">🏘️ {VILLAGE_NAME}便民政策AI助手</h1>
  <p style="margin:4px 0 0;font-size:14px;opacity:0.85;">{VILLAGE_FULL} · 大学生社会实践项目</p>
</div>
""", unsafe_allow_html=True)


# ====================================================================
#  UI: 双标签页
# ====================================================================

tab_qa, tab_map = st.tabs(["💬 政策问答", "🗺️ 村地图导览"])


# ########################
#  标签页1: 政策问答
# ########################

with tab_qa:

    # ----- 快捷问题 -----
    st.markdown("##### 📌 常见问题快速提问")
    qq_cols = st.columns(len(QUICK_QUESTIONS))
    for i, q in enumerate(QUICK_QUESTIONS):
        if qq_cols[i].button(q, key=f"qq_{i}", use_container_width=True):
            answer, sources = process_question(q)
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()

    # ----- 聊天消息展示 -----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.info(f"📄 **参考文件：** {'、'.join(msg['sources'])}")

    # ----- 聊天输入框 -----
    if prompt := st.chat_input(f"输入您的问题，如：医保怎么报销？"):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取 AI 回答
        with st.chat_message("assistant"):
            with st.status("🔍 正在查询政策文件...", expanded=False) as status:
                answer, sources = process_question(prompt)
            st.markdown(answer)
            if sources:
                st.info(f"📄 **参考文件：** {'、'.join(sources)}")
            status.update(label="✅ 回答完成", state="complete")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

    # ----- 清空对话 -----
    if st.session_state.messages and st.button("🗑️ 清空对话", type="secondary", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"对话已清空。您好！我是 **{VILLAGE_NAME}便民政策AI助手**，有什么政策问题尽管问我！",
            }
        ]
        st.rerun()

    # ----- 政策文件管理 -----
    with st.expander("📂 政策文件管理（村委使用）"):
        uploaded_files = st.file_uploader(
            "选择 .txt 或 .md 文件上传",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="policy_uploader",
        )

        if uploaded_files:
            POLICIES_DIR.mkdir(exist_ok=True)
            saved_count = 0
            for f in uploaded_files:
                file_path = POLICIES_DIR / f.name
                file_path.write_bytes(f.getvalue())
                saved_count += 1
                logger.info(f"新政策文件已保存: {file_path}")

            rebuild_vectorstore()
            st.success(f"✅ 成功上传 {saved_count} 个文件，知识库已更新！")
            st.rerun()

        # 显示已有政策文件
        if POLICIES_DIR.exists():
            policy_files = sorted(POLICIES_DIR.glob("*.txt")) + sorted(POLICIES_DIR.glob("*.md"))
            if policy_files:
                st.markdown("**📄 已有政策文件：**")
                file_display = ", ".join([
                    f"`{f.name}` ({f.stat().st_size // 1024}KB)" for f in policy_files
                ])
                st.markdown(file_display)
            else:
                st.info("📭 暂无政策文件，请上传或添加到 policies/ 目录。")


# ########################
#  标签页2: 村地图导览
# ########################

with tab_map:

    # ----- 导览说明 -----
    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:20px;margin-bottom:15px;
         box-shadow:0 2px 12px rgba(0,0,0,0.08);">
      <h2 style="font-size:24px;color:#1a73e8;margin-bottom:8px;">🗺️ {VILLAGE_NAME}地图导览</h2>
      <p style="font-size:16px;color:#666;line-height:1.6;">
        点击下方"导航到这里"按钮，即可打开高德地图导航至目的地。
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ----- 3D 高德地图（嵌入式） -----
    if "map_loaded" not in st.session_state:
        st.session_state.map_loaded = True

    map_html = f"""
    <!DOCTYPE html>
    <html><head>
    <meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;}}
      html,body{{width:100%;height:100%;background:transparent;font-family:sans-serif;}}
      #map_container{{width:100%;height:100%;border-radius:14px;overflow:hidden;background:#e8e4dc;}}
    </style>
    </head><body>
    <div id="search_box" style="position:absolute;top:12px;left:12px;right:12px;z-index:999;
         display:flex;gap:6px;background:rgba(255,255,255,0.95);padding:6px 6px 6px 14px;
         border-radius:24px;box-shadow:0 2px 10px rgba(0,0,0,0.15);backdrop-filter:blur(4px);">
      <span style="font-size:18px;line-height:36px;">🔍</span>
      <input id="search_input" type="text" placeholder="搜索目的地，如：村委会、卫生所..."
             style="flex:1;border:none;outline:none;font-size:16px;background:transparent;
                    padding:6px 0;line-height:24px;">
      <button id="search_btn" style="background:#1a73e8;color:#fff;border:none;
             padding:8px 22px;border-radius:20px;font-size:15px;font-weight:600;cursor:pointer;
             white-space:nowrap;">搜索</button>
    </div>
    <div id="map_container" style="width:100%;height:100%;border-radius:14px;overflow:hidden;background:#e8e4dc;"></div>
    <div id="map_error" style="display:none;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#666;font-size:14px;padding:20px;z-index:999;">
      ⚠️ 地图加载失败，请检查网络连接</div>
    <script src="https://webapi.amap.com/maps?v=2.0&key=7737390acdd3346faa50a63dfb1b7a41"></script>
    <script>
    (function() {{
        var errorDiv = document.getElementById('map_error');
        function tryInit(times) {{
            if (typeof AMap === 'undefined') {{
                errorDiv.textContent = '地图API加载失败，请刷新页面';
                errorDiv.style.display = 'block';
                return;
            }}
            var container = document.getElementById('map_container');
            if (!container || container.offsetHeight < 10) {{
                if (times > 0) {{
                    setTimeout(function(){{ tryInit(times - 1); }}, 100);
                }} else {{
                    errorDiv.textContent = '地图容器尺寸异常，请刷新页面';
                    errorDiv.style.display = 'block';
                }}
                return;
            }}
            try {{
                var locs = [
                    {{"name":"李营村委会","desc":"村委办公地点，办理盖章、证明、申请等各类村级事务","lat":33.063842,"lng":112.165433,"icon":"🏛️"}},
                    {{"name":"村卫生所","desc":"医保定点门诊，常见病诊疗、慢病取药","lat":33.062273,"lng":112.170341,"icon":"🏥"}},
                    {{"name":"党群服务中心","desc":"党组织和党员活动阵地，提供党务、政务、便民服务","lat":33.063785,"lng":112.165412,"icon":"⭐"}},
                    {{"name":"文化活动广场","desc":"村内活动集会场地，政策宣传公示栏所在地","lat":33.063978,"lng":112.165749,"icon":"🏟️"}},
                    {{"name":"便民快递超市","desc":"快递代收代寄，日常生活用品购买","lat":33.063321,"lng":112.169944,"icon":"🏪"}},
                ];
                var map = new AMap.Map('map_container', {{
                    zoom: 16,
                    center: [112.1674, 33.0634],
                    viewMode: '3D',
                    pitch: 55,
                    rotation: -15,
                    showIndoorMap: false,
                }});
                AMap.plugin(['AMap.ToolBar', 'AMap.Scale'], function() {{
                    map.addControl(new AMap.ToolBar({{position:'RT'}}));
                    map.addControl(new AMap.Scale());
                }});
                locs.forEach(function(loc) {{
                    var marker = new AMap.Marker({{
                        position: [loc.lng, loc.lat],
                        title: loc.name,
                        label: {{
                            content: '<div style="background:#1a73e8;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,0.2);">'+loc.name+'</div>',
                            direction: 'top',
                            offset: new AMap.Pixel(0, -6),
                        }},
                    }});
                    marker.on('click', function() {{
                        var navUrl = 'https://uri.amap.com/navigation?to='+loc.lng+','+loc.lat+','+encodeURIComponent(loc.name)+'&mode=car&coordinate=gaode';
                        var info = new AMap.InfoWindow({{
                            content: '<div style="padding:10px;font-size:14px;line-height:1.6;max-width:220px;">'+
                                '<div style="font-size:18px;font-weight:700;color:#222;margin-bottom:4px;">'+loc.name+'</div>'+
                                '<div style="color:#666;font-size:13px;margin-bottom:10px;">'+loc.desc+'</div>'+
                                '<a onclick="window.top.location.href=\''+navUrl+'\';return false;" style="display:inline-block;padding:8px 20px;background:#1a73e8;color:#fff;border-radius:20px;text-decoration:none;font-size:14px;font-weight:600;cursor:pointer;">📍 导航到这里</a>'+
                                '</div>',
                            offset: new AMap.Pixel(0, -28),
                        }});
                        info.open(map, marker.getPosition());
                    }});
                    map.add(marker);
                }});
                // 移动端适配
                setTimeout(function(){{ map.resize(); }}, 800);
                window.addEventListener("resize", function(){{ map.resize(); }});
                // 搜索功能
                var searchInput = document.getElementById('search_input');
                var searchBtn = document.getElementById('search_btn');
                function doSearch() {{
                    var keyword = searchInput.value.trim();
                    if (!keyword) return;
                    var matched = null;
                    for (var i = 0; i < locs.length; i++) {{
                        if (locs[i].name.indexOf(keyword) > -1 || locs[i].desc.indexOf(keyword) > -1) {{
                            matched = locs[i];
                            break;
                        }}
                    }}
                    if (matched) {{
                        map.setCenter([matched.lng, matched.lat]);
                        map.setZoom(18);
                        return;
                    }}
                    AMap.plugin('AMap.Geocoder', function() {{
                        var geocoder = new AMap.Geocoder({{city:'镇平县',radius:1000}});
                        geocoder.getLocation(keyword, function(status, result) {{
                            if (status === 'complete' && result.geocodes.length) {{
                                var lnglat = result.geocodes[0].getLocation();
                                map.setCenter(lnglat);
                                map.setZoom(17);
                                new AMap.Marker({{position:lnglat,title:keyword}});
                            }} else {{
                                alert('未找到 "'+keyword+'" 相关位置');
                            }}
                        }});
                    }});
                }}
                searchBtn.addEventListener('click', doSearch);
                searchInput.addEventListener('keydown', function(e) {{ if(e.keyCode===13) doSearch(); }});
            }} catch(e) {{
                errorDiv.textContent = '地图加载失败: ' + (e.message || '未知错误');
                errorDiv.style.display = 'block';
            }}
        }}
        // 延迟50ms开始尝试，最多50次(5秒)
        setTimeout(function(){{ tryInit(50); }}, 50);
    }})();
    </script>
    </body></html>
    """

    st.components.v1.html(map_html, height=510)

    # ----- 地点卡片（2列网格） -----
    for i in range(0, len(VILLAGE_LOCATIONS), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(VILLAGE_LOCATIONS):
                loc = VILLAGE_LOCATIONS[idx]
                nav_url = (
                    f"https://uri.amap.com/navigation?"
                    f"to={loc['lng']},{loc['lat']},{loc['name']}"
                    f"&mode=car&coordinate=gaode"
                )
                with cols[j]:
                    st.markdown(f"""
                    <div style="background:white;border-radius:14px;padding:18px;
                         box-shadow:0 2px 10px rgba(0,0,0,0.08);margin-bottom:12px;
                         transition:transform 0.15s;">
                      <div style="font-size:40px;margin-bottom:6px;">{loc.get('icon', '📍')}</div>
                      <div style="font-size:22px;font-weight:700;color:#222;margin-bottom:4px;">
                        {loc['name']}</div>
                      <div style="font-size:15px;color:#666;line-height:1.6;margin-bottom:12px;
                           min-height:2.4em;">
                        {loc['desc']}</div>
                      <a href="{nav_url}" target="_blank" rel="noopener"
                         style="text-decoration:none;">
                        <button style="width:100%;padding:12px 24px;background:#1a73e8;color:white;
                             border:none;border-radius:25px;font-size:17px;font-weight:600;
                             cursor:pointer;transition:background 0.2s;
                             -webkit-tap-highlight-color:transparent;"
                         onmouseover="this.style.background='#1557b0'"
                         onmouseout="this.style.background='#1a73e8'">
                          📍 导航到这里
                        </button>
                      </a>
                    </div>
                    """, unsafe_allow_html=True)

    # ----- 提示信息 -----
    st.info(
        "📱 点击导航按钮将自动打开高德地图（手机端）或高德地图网页版（电脑端）。\n\n"
        "坐标如有偏差，请在实际使用中校准。"
    )


# ====================================================================
#  UI: 底部备案信息（显示在两个标签页下方）
# ====================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:20px 10px;">
  <div style="color:#1a73e8;font-size:16px;font-weight:600;letter-spacing:1px;">
    大学生社会实践助力乡村振兴 🌾</div>
  <div style="color:#888;font-size:13px;margin-top:6px;">
    本服务已在 {VILLAGE_FULL} 备案</div>
  <div style="color:#aaa;font-size:12px;margin-top:4px;">
    （以上信息仅供参考，具体政策以村委会解释为准）</div>
</div>
""", unsafe_allow_html=True)
