import os
import json
import re
import dotenv
from typing import TypedDict, List, Dict, Optional, Any, Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import pprint
import networkx as nx
from utils import get_llm, get_evaluation_llm, get_llm_user


# ================================================================= #
# 1. 环境与模型设置 & 图谱路径工具
# ================================================================= #

def get_graph_path():
    """获取知识图谱(NetworkX)的文件路径"""
    dotenv.load_dotenv()
    return os.path.join(os.getenv("MEMORY_ROOT"), os.getenv("CURRENT_PROJECT_ID"), "story_graph.json")


def load_graph() -> nx.Graph:
    """加载图谱 (修复了 FutureWarning)"""
    graph_path = get_graph_path()
    if os.path.exists(graph_path):
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 显式指定 edges="links" 以消除警告
            return nx.node_link_graph(data, edges="links")
        except Exception as e:
            print(f"⚠️ 读取图谱失败，将创建新图: {e}")
            return nx.Graph()
    else:
        return nx.Graph()


def save_graph(G: nx.Graph):
    """保存图谱 (修复了 FutureWarning)"""
    graph_path = get_graph_path()
    # 显式指定 edges="links" 以消除警告
    data = nx.node_link_data(G, edges="links")
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ... (第2部分：数据结构定义保持不变，省略以节省空间) ...
# ... (RelationshipStatus, CharacterProfile, KnowledgeBase, Pydantic模型等) ...
RelationshipStatus = Literal[
    "朋友", "恋人", "敌人", "陌生人", "家庭成员", "合作伙伴", "竞争对手", "初次见面", "合租室友"]


class CharacterProfile(TypedDict):
    """扩展的角色档案 (已更新关系存储结构)"""
    name: str
    age: Optional[int]
    gender: Optional[str]
    backstory: str
    traits: List[str]  # 性格特点
    specialties: List[str]
    hobbies: List[str]
    occupations: List[str]
    appearance: str  # 外貌描述
    # --- 结构更新 ---
    # 使用字典存储，键为对方角色名，值为关系详情
    relationship: Dict[str, Dict[str, str]]
    first_appearance_chapter: int  # 首次出现的章节
    aliases: List[str]  # 新增别名


class KnowledgeBase(TypedDict):
    """知识库的顶层结构"""
    characters: Dict[str, CharacterProfile]
    last_updated_chapter: int


# Pydantic 模型用于输出解析
class CharacterInfo(BaseModel):
    """角色信息模型"""
    name: str = Field(description="角色姓名")
    estimated_age: str = Field(description="数字/未知")
    gender: str = Field(description="男/女/未知")
    appearance: str = Field(description="外貌描述/未知")
    traits: List[str] = Field(description="性格特点列表")
    specialties: List[str] = Field(description="特长列表")
    hobbies: List[str] = Field(description="爱好列表")
    occupations: List[str] = Field(description="职业列表")
    aliases: List[str] = Field(description="本章中出现的该角色的其他称呼/别名", default=[])


class CharacterInteraction(BaseModel):
    """角色互动模型"""
    character_a: str = Field(description="角色A姓名")
    character_b: str = Field(description="角色B姓名")
    interaction_type: str = Field(description="关系类型")
    interaction_summary: str = Field(description="互动描述")


class CharacterAnalysisResult(BaseModel):
    """角色分析结果模型"""
    characters: List[CharacterInfo] = Field(description="角色列表")
    character_interactions: List[CharacterInteraction] = Field(description="角色互动列表")


class EvidenceText(BaseModel):
    """证据文本模型"""
    evidence: str = Field(description="从章节内容中提取的互动文本片段")


class BackstorySummary(BaseModel):
    """背景故事摘要模型"""
    summary: str = Field(description="角色个人背景故事的总结")


# ================================================================= #
# 3. 基础工具集 (初始化、读取、保存)
# ================================================================= #

def initialize_knowledge_base(KNOWLEDGE_BASE_PATH: str):
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        initial_kb = {"characters": {}, "last_updated_chapter": 0}
        with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
            json.dump(initial_kb, f, ensure_ascii=False, indent=2)
        print("✅ 已初始化知识库文件")


def get_knowledge_base(KNOWLEDGE_BASE_PATH: str) -> KnowledgeBase:
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        if "characters" not in kb: kb["characters"] = {}
        if "last_updated_chapter" not in kb: kb["last_updated_chapter"] = 0
        for char_name in kb["characters"]:
            if "relationship" not in kb["characters"][char_name]:
                kb["characters"][char_name]["relationship"] = {}
            if "aliases" not in kb["characters"][char_name]:
                kb["characters"][char_name]["aliases"] = []
        return kb
    except FileNotFoundError:
        initialize_knowledge_base(KNOWLEDGE_BASE_PATH)
        return get_knowledge_base(KNOWLEDGE_BASE_PATH)


def save_knowledge_base(kb: KnowledgeBase, KNOWLEDGE_BASE_PATH: str):
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)


def clean_llm_response(content: str) -> str:
    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json"):-len("```")].strip()
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match: content = json_match.group().strip()
    content = content.replace("'", '\"')
    return re.sub(r"//.*|/\*[\s\S]*?\*/", "", content)


# ================================================================= #
# 4. 核心逻辑函数 (修复版)
# ================================================================= #

def identify_characters_in_text(text: str, current_chapter: int, narrator_name: str = None) -> Dict[str, Any]:
    """第一步：识别文本中出现的主要角色"""

    # 获取已知角色列表 (Entity Registry)
    dotenv.load_dotenv()
    knowledge_path = os.path.join(os.getenv("MEMORY_ROOT"), os.getenv("CURRENT_PROJECT_ID"), "knowledge_base.json")
    existing_chars_desc = ""
    try:
        kb = get_knowledge_base(knowledge_path)
        if kb.get("characters"):
            existing_chars_desc = "### 已知角色注册表 (请优先将文中人物映射到以下标准名):\n"
            for name, data in kb["characters"].items():
                aliases = data.get("aliases", [])
                desc = data.get("traits", [])[:3]
                existing_chars_desc += f"- 标准名: {name} | 已知别名: {aliases} | 特征: {desc}\n"
    except:
        pass

    # 主角映射逻辑
    narrator_instruction = ""
    if narrator_name:
        narrator_instruction = f"""
6. **强制指代消解**：本章是以第一人称叙述的。文中的 **"我"** 指代的是主角 **"{narrator_name}"**。
   - 输出时请直接使用标准名 "{narrator_name}"，**不要** 输出 "我"。
"""

    parser = PydanticOutputParser(pydantic_object=CharacterAnalysisResult)
    prompt_template = f"""
任务：分析以下小说章节，识别所有主要角色及他们的互动关系。

{existing_chars_desc}

### 章节内容
第{{current_chapter}}章：
{{text}}

### 输出要求
1. 如果未提及属性可用'未知'代替。
2. traits, hobbies等可是多个值。
3. **泛称处理**：对于"一群科学家"、"路人"等非具体群体，除非是关键角色，否则**不要**提取。
{narrator_instruction}

{{format_instructions}}
"""
    master_llm = get_evaluation_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text", "current_chapter"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | master_llm
        raw_response = chain.invoke({"text": text, "current_chapter": current_chapter})

        from helper import filter_think_tags, parse_json_with_filtering
        filtered_content = filter_think_tags(raw_response.content)
        parsed_result = parse_json_with_filtering(filtered_content, parser)

        result = {
            "characters": [char.model_dump() for char in parsed_result.characters],
            "character_interactions": [interaction.model_dump() for interaction in parsed_result.character_interactions]
        }
        print(f"✅ 角色识别成功：{len(result['characters'])}个角色，{len(result['character_interactions'])}次互动")
        return result
    except Exception as e:
        print(f"--- ❌ 角色识别解析失败: {e} ---")
        return {"characters": [], "character_interactions": []}


def create_character_if_not_exists(name: str, age: Optional[str] = None,
                                   gender: Optional[str] = None,
                                   appearance: str = "", traits: List[str] = None,
                                   specialties: List[str] = None, hobbies: List[str] = None,
                                   occupations: List[str] = None, current_chapter: int = 0,
                                   relationship: Dict[str, Dict[str, str]] = None,
                                   aliases: List[str] = None) -> str:
    """第二步：创建角色（同步更新图谱）"""
    dotenv.load_dotenv()
    knowledge_path = os.path.join(os.getenv("MEMORY_ROOT"), os.getenv("CURRENT_PROJECT_ID"), "knowledge_base.json")
    kb = get_knowledge_base(knowledge_path)
    traits = traits or []
    relationship = relationship or {}
    aliases = aliases or []

    msg = ""
    # 1. 更新 JSON
    if name not in kb["characters"]:
        kb["characters"][name] = {
            "name": name, "age": age, "gender": gender, "backstory": "",
            "traits": traits, "appearance": appearance, "relationship": relationship,
            "first_appearance_chapter": current_chapter, "specialties": specialties,
            "hobbies": hobbies, "occupations": occupations, "aliases": aliases
        }
        msg = f"✅ 成功创建角色 '{name}'"
    else:
        char = kb["characters"][name]
        if "aliases" not in char: char["aliases"] = []
        new_aliases = [a for a in aliases if a not in char["aliases"] and a != name]
        if new_aliases: char["aliases"].extend(new_aliases)
        msg = f"✅ 角色 '{name}' 已存在"

    save_knowledge_base(kb, knowledge_path)

    # 2. 更新 Graph
    try:
        G = load_graph()
        if not G.has_node(name): G.add_node(name)
        node_attrs = {"age": age, "gender": gender, "traits": traits, "id": name}
        for k, v in node_attrs.items():
            if v: G.nodes[name][k] = v
        save_graph(G)
    except Exception as e:
        print(f"⚠️ 图谱节点更新异常: {e}")

    return msg


# ... (update_character_backstory 和 update_relationship 保持不变，注意 load_graph 已更新) ...
def update_character_backstory(name: str, new_information: str,
                               mode: Literal['append', 'overwrite'] = 'append',
                               current_chapter: Optional[int] = None) -> str:
    """更新角色个人背景故事"""
    dotenv.load_dotenv()
    knowledge_path = os.path.join(os.getenv("MEMORY_ROOT"), os.getenv("CURRENT_PROJECT_ID"), "knowledge_base.json")
    kb = get_knowledge_base(knowledge_path)

    if name not in kb["characters"]: return f"❌ 角色 '{name}' 不存在"

    formatted_info = f"第{current_chapter}章：{new_information}" if current_chapter else new_information
    char = kb["characters"][name]

    if mode == 'append':
        char["backstory"] = (char["backstory"] + f"\n{formatted_info}") if char["backstory"] else formatted_info
    elif mode == 'overwrite':
        char["backstory"] = formatted_info

    save_knowledge_base(kb, knowledge_path)
    return f"✅ 更新背景故事: {name}"


def update_relationship(character_a: str, character_b: str,
                        new_status: str,
                        event_summary: str, chapter_evidence: str,
                        current_chapter: int) -> str:
    """第四步：更新角色关系（同步更新图谱）"""
    dotenv.load_dotenv()
    knowledge_path = os.path.join(os.getenv("MEMORY_ROOT"), os.getenv("CURRENT_PROJECT_ID"), "knowledge_base.json")
    kb = get_knowledge_base(knowledge_path)

    if character_a not in kb["characters"] or character_b not in kb["characters"]:
        return f"❌ 错误：角色不存在 (请确保 {character_a} 和 {character_b} 都已创建)"

    # 1. 更新常规知识库
    current_relationship_dict = kb["characters"][character_a]["relationship"].get(character_b, {})
    current_status = next(iter(current_relationship_dict.keys()), None)

    if new_status != current_status:
        relationship_entry = {new_status: f"{event_summary} (第{current_chapter}章)"}
        kb["characters"][character_a]["relationship"][character_b] = relationship_entry
        kb["characters"][character_b]["relationship"][character_a] = relationship_entry
        save_knowledge_base(kb, knowledge_path)

        # 2. 同步更新 NetworkX 知识图谱
        try:
            G = load_graph()
            if not G.has_node(character_a): G.add_node(character_a)
            if not G.has_node(character_b): G.add_node(character_b)
            G.add_edge(character_a, character_b,
                       relation=new_status,
                       summary=event_summary,
                       chapter=current_chapter)
            save_graph(G)
        except Exception as e:
            print(f"⚠️ 图谱关系更新异常: {e}")

        return f"✅ 关系更新：{character_a} ↔ {character_b}（{new_status}）"
    else:
        return f"ℹ️  关系未变，跳过更新"


# ... (detect_personal_backstory_updates, analyze_relationship_changes 保持不变) ...
def detect_personal_backstory_updates(characters: List[Dict], chapter_text: str, current_chapter: int) -> List[Dict]:
    """检测并生成个人背景故事更新任务"""
    tool_calls = []
    parser = PydanticOutputParser(pydantic_object=BackstorySummary)
    prompt_template = """
任务：分析以下章节内容，提取角色'{name}'的个人背景故事信息。
章节内容：{chapter_text}
要求：简短总结（<100字），无信息返"无"。
{format_instructions}
"""
    master_llm = get_evaluation_llm()

    for char in characters:
        name = char.get("name", "").strip()
        if not name: continue
        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["name", "chapter_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            chain = prompt | master_llm
            raw_response = chain.invoke({"name": name, "chapter_text": chapter_text[:1500]})
            from helper import filter_think_tags, parse_json_with_filtering
            filtered_content = filter_think_tags(raw_response.content)
            parsed_summary = parse_json_with_filtering(filtered_content, parser)
            summary = parsed_summary.summary.strip()
        except Exception:
            summary = "无"

        if summary and summary != "无":
            tool_calls.append({
                "name": "update_character_backstory",
                "args": {"name": name, "new_information": summary, "mode": "append", "current_chapter": current_chapter}
            })
    return tool_calls


def analyze_relationship_changes(character_interactions: List[Dict], chapter_text: str, current_chapter: int) -> List[
    Dict]:
    """第三步：分析关系变化"""
    tool_calls = []
    if not character_interactions: return tool_calls

    parser = PydanticOutputParser(pydantic_object=EvidenceText)
    prompt_template = """
任务：提取角色'{char_a}'和'{char_b}'互动的具体文本片段。
章节内容：{chapter_text}
{format_instructions}
"""
    master_llm = get_evaluation_llm()

    for interaction in character_interactions:
        char_a = interaction.get("character_a", "").strip()
        char_b = interaction.get("character_b", "").strip()
        interaction_type = interaction.get("interaction_type", "陌生人").strip()
        summary = interaction.get("interaction_summary", "无描述").strip()

        if not char_a or not char_b: continue

        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["char_a", "char_b", "chapter_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            chain = prompt | master_llm
            raw_response = chain.invoke({
                "char_a": char_a, "char_b": char_b, "chapter_text": chapter_text[:1000]
            })
            from helper import filter_think_tags, parse_json_with_filtering
            filtered_content = filter_think_tags(raw_response.content)
            parsed_evidence = parse_json_with_filtering(filtered_content, parser)
            evidence = parsed_evidence.evidence.strip() or "未提取"
        except Exception:
            evidence = "未提取"

        tool_calls.append({
            "name": "update_relationship",
            "args": {
                "character_a": char_a, "character_b": char_b,
                "new_status": interaction_type, "event_summary": summary,
                "chapter_evidence": evidence, "current_chapter": current_chapter
            }
        })
    return tool_calls


def run_complete_relationship_analysis(chapter_text: str, current_chapter: int = 1):
    """完整的角色和关系分析流程 (修复版)"""
    print(f"\n{'=' * 50}\n🔍 开始分析第{current_chapter}章\n{'=' * 50}")

    # 1. 识别角色
    identification_result = identify_characters_in_text(chapter_text, current_chapter)
    characters = identification_result.get("characters", [])
    interactions = identification_result.get("character_interactions", [])

    character_tool_calls = []
    # 记录已经准备创建的角色名，防止重复
    chars_to_create = set()

    # 2. 处理LLM明确识别出的角色
    for char in characters:
        name = char.get("name", "").strip()
        if name:
            chars_to_create.add(name)
            character_tool_calls.append({
                "name": "create_character_if_not_exists",
                "args": {
                    "name": name,
                    "age": char.get("estimated_age"),
                    "gender": char.get("gender"),
                    "appearance": (char.get("appearance") or "").strip(),
                    "traits": char.get("traits", []),
                    "specialties": char.get("specialties", []),
                    "hobbies": char.get("hobbies", []),
                    "occupations": char.get("occupations", []),
                    "aliases": char.get("aliases", []),
                    "current_chapter": current_chapter,
                    "relationship": {}
                }
            })

    # 3. 【关键修复】检查互动中是否有未创建的角色，自动补全
    for interaction in interactions:
        for key in ["character_a", "character_b"]:
            char_name = interaction.get(key, "").strip()
            if char_name and char_name not in chars_to_create:
                print(f"⚠️ 自动补全漏网角色创建任务: {char_name}")
                chars_to_create.add(char_name)
                character_tool_calls.append({
                    "name": "create_character_if_not_exists",
                    "args": {
                        "name": char_name,
                        "age": "未知", "gender": "未知", "appearance": "未知",
                        "traits": [], "specialties": [], "hobbies": [], "occupations": [], "aliases": [],
                        "current_chapter": current_chapter, "relationship": {}
                    }
                })

    print(f"📋 生成角色操作任务：{len(character_tool_calls)}个")

    # 4. 背景故事和关系更新
    backstory_tool_calls = detect_personal_backstory_updates(characters, chapter_text, current_chapter)
    relationship_tool_calls = analyze_relationship_changes(interactions, chapter_text, current_chapter)

    # 必须保证 character_tool_calls 在最前面执行
    all_tool_calls = character_tool_calls + backstory_tool_calls + relationship_tool_calls
    print(f"\n✅ 分析完成：共生成 {len(all_tool_calls)} 个操作任务")
    return all_tool_calls


# ... (execute_tool_calls, simulate_user_confirmation 等保持原样) ...
def execute_tool_calls(tool_calls: List[Dict]):
    """执行工具调用"""
    available_tools = {
        "create_character_if_not_exists": create_character_if_not_exists,
        "update_character_backstory": update_character_backstory,
        "update_relationship": update_relationship,
        "merge_characters": merge_characters  # 注册合并工具
    }
    results = []
    if not tool_calls: return results

    print(f"\n🛠️  开始执行 {len(tool_calls)} 个工具调用...")
    for idx, call in enumerate(tool_calls, 1):
        print(f"\n--- 任务{idx}/{len(tool_calls)}：{call['name']} ---")
        if call["name"] not in available_tools:
            print(f"❌ 未知工具：{call['name']}")
            continue
        try:
            tool_func = available_tools[call["name"]]
            result = tool_func(**call["args"])
            print(f"✅ 结果：{result}")
            results.append(result)
        except Exception as e:
            print(f"❌ 执行失败：{str(e)}")
    return results


def simulate_user_confirmation_and_execute(tool_calls: List[Dict]):
    # ... (与原代码一致) ...
    if not tool_calls:
        print("ℹ️  无待执行操作")
        return []
    return execute_tool_calls(tool_calls)


# ================================================================= #
# 5. 图谱推理与合并工具 (新加)
# ================================================================= #

def merge_characters(primary_name: str, alias_name: str) -> str:
    """合并两个实体节点 (例如 '我' -> '林夏')"""
    print(f"--- 🔄 开始合并实体: '{alias_name}' -> '{primary_name}' ---")
    dotenv.load_dotenv()
    kb_path = os.path.join(os.getenv("MEMORY_ROOT"), os.getenv("CURRENT_PROJECT_ID"), "knowledge_base.json")

    try:
        kb = get_knowledge_base(kb_path)
        if primary_name not in kb["characters"]: return f"❌ 主角色 '{primary_name}' 不存在"
        if alias_name not in kb["characters"]: return f"ℹ️  别名角色 '{alias_name}' 不存在，跳过"

        p_data = kb["characters"][primary_name]
        a_data = kb["characters"][alias_name]

        # 合并属性
        for field in ["traits", "specialties", "hobbies", "occupations"]:
            if a_data.get(field):
                p_data[field] = list(set(p_data.get(field, []) + a_data[field]))

        # 合并背景
        if a_data.get("backstory"):
            p_data["backstory"] += f"\n【来自{alias_name}】：{a_data['backstory']}"

        # 合并关系
        if "relationship" in a_data:
            for target, rel in a_data["relationship"].items():
                if target == primary_name: continue
                if target not in p_data["relationship"]:
                    p_data["relationship"][target] = rel
                    # 反向更新对方的关系指向
                    if target in kb["characters"] and alias_name in kb["characters"][target]["relationship"]:
                        old_rel = kb["characters"][target]["relationship"].pop(alias_name)
                        kb["characters"][target]["relationship"][primary_name] = old_rel

        # 记录别名
        if "aliases" not in p_data: p_data["aliases"] = []
        if alias_name not in p_data["aliases"]: p_data["aliases"].append(alias_name)

        del kb["characters"][alias_name]
        save_knowledge_base(kb, kb_path)

        # 图谱合并
        G = load_graph()
        if G.has_node(alias_name):
            if not G.has_node(primary_name): G.add_node(primary_name, **G.nodes[alias_name])
            for n in list(G.neighbors(alias_name)):
                if n == primary_name: continue
                if not G.has_edge(primary_name, n):
                    G.add_edge(primary_name, n, **G.get_edge_data(alias_name, n))
            G.remove_node(alias_name)
            save_graph(G)

        return f"✅ 成功合并: {alias_name} -> {primary_name}"
    except Exception as e:
        return f"❌ 合并出错: {e}"


def get_story_graph_context(focus_characters: List[str]) -> str:
    """图谱推理引擎"""
    try:
        G = load_graph()
        if G.number_of_nodes() == 0: return "暂无图谱数据。"
        valid_chars = [c for c in focus_characters if G.has_node(c)]
        if not valid_chars: return "无相关图谱信息。"

        context = []
        if len(valid_chars) == 1:
            char = valid_chars[0]
            context.append(f"### 【{char}】的社交圈")
            for n in G.neighbors(char):
                ed = G.get_edge_data(char, n)
                context.append(f"- {n} ({ed.get('relation', '')}): {ed.get('summary', '')}")
        else:
            context.append("### 人物关系推理")
            subgraph = G.subgraph(valid_chars)
            for u, v, d in subgraph.edges(data=True):
                context.append(f"- {u}<->{v}: {d.get('relation')} ({d.get('summary')})")
        return "\n".join(context)
    except:
        return "图谱推理服务暂不可用。"


if __name__ == "__main__":
    pass