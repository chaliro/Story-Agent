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
from utils import get_llm, get_evaluation_llm, get_llm_user

# ================================================================= #
# 1. 环境与模型设置
# ================================================================= #

# ================================================================= #
# 2. 扩展的知识库结构定义 (已更新) & Pydantic 模型
# ================================================================= #

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
# 3. 改进的工具集 (核心更新) - 添加输出解析功能
# ================================================================= #

def initialize_knowledge_base(KNOWLEDGE_BASE_PATH: str):
    """初始化知识库文件"""
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        initial_kb = {
            "characters": {},
            "last_updated_chapter": 0
        }
        with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
            json.dump(initial_kb, f, ensure_ascii=False, indent=2)
        print("✅ 已初始化知识库文件")


def get_knowledge_base(KNOWLEDGE_BASE_PATH: str) -> KnowledgeBase:
    """获取当前知识库，并确保其结构完整性"""
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        # 确保所有必需的字段都存在
        if "characters" not in kb:
            kb["characters"] = {}
            print("⚠️  知识库文件缺少 'characters' 字段，已自动修复。")

        if "last_updated_chapter" not in kb:
            kb["last_updated_chapter"] = 0
            print("⚠️  知识库文件缺少 'last_updated_chapter' 字段，已自动修复。")

        # 【重要】为旧版知识库文件添加新的 'relationship' 字段
        for char_name in kb["characters"]:
            if "relationship" not in kb["characters"][char_name]:
                kb["characters"][char_name]["relationship"] = {}
                print(f"⚠️  为角色 '{char_name}' 自动添加 'relationship' 字段。")

        # 如果进行了修复，保存更改
        if any("relationship" not in char for char in kb["characters"].values()):
            dotenv.load_dotenv()
            dir1 = os.getenv("MEMORY_ROOT")
            dir2 = os.getenv("CURRENT_PROJECT_ID")

            dir5 = "knowledge_base.json"

            knowledge_path = os.path.join(dir1, dir2, dir5)
            save_knowledge_base(kb, knowledge_path)

        return kb

    except FileNotFoundError:
        initialize_knowledge_base(KNOWLEDGE_BASE_PATH)
        return get_knowledge_base(KNOWLEDGE_BASE_PATH)


def save_knowledge_base(kb: KnowledgeBase, KNOWLEDGE_BASE_PATH: str):
    """保存知识库"""
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)


def clean_llm_response(content: str) -> str:
    """清洗LLM返回的内容，提取纯净JSON"""
    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json"):-len("```")].strip()
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        content = json_match.group().strip()
    content = content.replace("'", '\"')
    content = re.sub(r"//.*", "", content)
    content = re.sub(r"/\*[\s\S]*?\*/", "", content)
    return content


def identify_characters_in_text(text: str, current_chapter: int) -> Dict[str, Any]:
    """第一步：识别文本中出现的主要角色 - 使用输出解析器版本"""
    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=CharacterAnalysisResult)

    prompt_template = """
任务：分析以下小说章节，识别所有主要角色及他们的互动关系。

### 章节内容
第{current_chapter}章：
{text}

### 输出要求
1. 如果未提及estimated_age、gender、appearance，可用未知代替，不要自己猜一个结果
2. traits，hobbies，occupations，specialties若没有提交也可以是未知，若原文有设计或者暗示，可以添加相应内容，这几个属性都可以是未知，或者只有1个值或者多个值都可以

{format_instructions}
"""

    master_llm = get_evaluation_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text", "current_chapter"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | master_llm
        raw_response = chain.invoke({
            "text": text,
            "current_chapter": current_chapter
        })

        # 过滤思考标签并解析
        from helper import filter_think_tags, parse_json_with_filtering
        filtered_content = filter_think_tags(raw_response.content)
        parsed_result = parse_json_with_filtering(filtered_content, parser)

        # 转换为字典格式以保持向后兼容
        result = {
            "characters": [char.model_dump() for char in parsed_result.characters],
            "character_interactions": [interaction.model_dump() for interaction in parsed_result.character_interactions]
        }

        print(f"✅ 角色识别成功：{len(result['characters'])}个角色，{len(result['character_interactions'])}次互动")
        return result

    except Exception as e:
        print(f"--- ❌ 角色识别输出解析失败: {e} ---")
        print("--- 🔄 使用传统解析方法作为回退 ---")
        return identify_characters_in_text_fallback(text, current_chapter)


def identify_characters_in_text_fallback(text: str, current_chapter: int) -> Dict[str, Any]:
    """回退方法：使用传统的JSON解析方式"""
    prompt = f"""
任务：分析以下小说章节，识别所有主要角色及他们的互动关系。

### 章节内容
第{current_chapter}章：
{text}

### 输出要求（必须严格遵守）
1. 仅返回标准JSON，不要包含任何解释性文字、代码块标记。
2. JSON语法必须正确：键名用双引号、字符串用双引号、无尾逗号。
3.如果未提及estimated_age、gender、appearance，可用未知代替，不要自己猜一个结果
4.traits，hobbies，occupations，specialties若没有提交也可以是未知，若原文有设计或者暗示，可以添加相应内容，这几个属性都可以是未知，或者只有1个值或者多个值都可以
5. JSON固定结构如下：
{{
    "characters": [
        {{
            "name": "角色姓名",
            "estimated_age": "数字/未知",
            "gender": "男/女/未知",
            "appearance": "外貌描述/未知",
            "traits": ["性格特点1", "性格特点2"],
            "specialties":["特长1", "特长2"]
            "hobbies":["爱好1", "爱好2"]
            "occupations":["职业1", "职业2"]
        }}
    ],
    "character_interactions": [
        {{
            "character_a": "角色A姓名",
            "character_b": "角色B姓名",
            "interaction_type": "关系类型（从['朋友','恋人','敌人','陌生人','初次见面', '合租室友']中选）",
            "interaction_summary": "互动描述"
        }}
    ]
}}
"""
    master_llm = get_evaluation_llm()
    response = master_llm.invoke(prompt)
    cleaned_content = clean_llm_response(response.content)
    try:
        result = json.loads(cleaned_content)
        if "characters" not in result:
            result["characters"] = []
        if "character_interactions" not in result:
            result["character_interactions"] = []
        print(
            f"✅ 角色识别成功（回退方法）：{len(result['characters'])}个角色，{len(result['character_interactions'])}次互动")
        return result
    except json.JSONDecodeError as e:
        print("\n❌ 角色识别JSON解析失败！")
        print(f"📄 LLM原始响应：\n{response.content[:500]}...")
        print(f"🧹 清洗后内容：\n{cleaned_content[:500]}...")
        print(f"💥 错误详情：{str(e)}")
        return {"characters": [], "character_interactions": []}


def create_character_if_not_exists(name: str, age: Optional[int] = None,
                                   gender: Optional[str] = None,
                                   appearance: str = "", traits: List[str] = None,
                                   specialties: List[str] = None, hobbies: List[str] = None,
                                   occupations: List[str] = None, current_chapter: int = 0,
                                   relationship: Dict[str, Dict[str, str]] = None) -> str:
    """第二步：检查并创建角色节点 (已更新)"""
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")

    dir5 = "knowledge_base.json"

    knowledge_path = os.path.join(dir1, dir2, dir5)
    kb = get_knowledge_base(knowledge_path)
    traits = traits or []
    relationship = relationship or {}  # 确保为字典

    if name not in kb["characters"]:
        kb["characters"][name] = {
            "name": name,
            "age": age,
            "gender": gender,
            "backstory": "",
            "traits": traits,
            "appearance": appearance,
            "relationship": relationship,  # 使用新键
            "first_appearance_chapter": current_chapter,
            "specialties": specialties,
            "hobbies": hobbies,
            "occupations": occupations
        }

        save_knowledge_base(kb, knowledge_path)
        return f"✅ 成功创建新角色 '{name}'"
    else:
        updated_fields = []
        character = kb["characters"][name]
        if age is not None and character["age"] is None:
            character["age"] = age
            updated_fields.append(f"年龄={age}")
        if gender is not None and character["gender"] is None:
            character["gender"] = gender
            updated_fields.append(f"性别={gender}")
        if appearance and not character["appearance"]:
            character["appearance"] = appearance
            updated_fields.append("外貌描述")
        for trait in traits:
            if trait not in character["traits"] and trait.strip():
                character["traits"].append(trait.strip())
                updated_fields.append(f"性格={trait}")
        for specialty in specialties:
            if specialty not in character["specialties"] and specialty.strip():
                character["specialties"].append(specialty.strip())
                updated_fields.append(f"特长={specialty}")
        for hobby in hobbies:
            if hobby not in character["hobbies"] and hobby.strip():
                character["hobbies"].append(hobby.strip())
                updated_fields.append(f"爱好={hobby}")
        for occupation in occupations:
            if occupation not in character["occupations"] and occupation.strip():
                character["occupations"].append(occupation.strip())
                updated_fields.append(f"职业={occupation}")

        if updated_fields:
            save_knowledge_base(kb, knowledge_path)
            return f"✅ 更新角色：{name}（更新字段：{', '.join(updated_fields)}）"
        else:
            return f"ℹ️  角色 '{name}' 已存在，无需更新"


def update_character_backstory(name: str, new_information: str,
                               mode: Literal['append', 'overwrite'] = 'append') -> str:
    """更新角色背景故事"""
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")

    dir5 = "knowledge_base.json"

    knowledge_path = os.path.join(dir1, dir2, dir5)
    kb = get_knowledge_base(knowledge_path)

    if name not in kb["characters"]:
        return f"❌ 错误：角色 '{name}' 不存在，无法更新背景故事"

    if mode == 'append':
        if kb["characters"][name]["backstory"]:
            kb["characters"][name]["backstory"] += f"\n{new_information}"
        else:
            kb["characters"][name]["backstory"] = new_information
    elif mode == 'overwrite':
        kb["characters"][name]["backstory"] = new_information

    save_knowledge_base(kb, knowledge_path)
    return f"✅ 成功更新角色 '{name}' 的背景故事"


def update_relationship(character_a: str, character_b: str,
                        new_status: RelationshipStatus,
                        event_summary: str, chapter_evidence: str,
                        current_chapter: int) -> str:
    """
    第四步：更新角色关系 (修复版本)
    仅在关系发生变化时更新，但不再自动更新背景故事
    """
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")

    dir5 = "knowledge_base.json"

    knowledge_path = os.path.join(dir1, dir2, dir5)
    kb = get_knowledge_base(knowledge_path)

    if character_a not in kb["characters"] or character_b not in kb["characters"]:
        return f"❌ 错误：角色 '{character_a}' 或 '{character_b}' 不存在"
    if character_a == character_b:
        return f"❌ 错误：不能添加角色与自身的关系"

    # 获取角色A当前与角色B的关系状态
    current_relationship_dict = kb["characters"][character_a]["relationship"].get(character_b, {})
    current_status = next(iter(current_relationship_dict.keys()), None)

    # --- 核心逻辑：仅在关系发生变化时执行 ---
    if new_status != current_status:
        # 构建新的关系条目
        relationship_entry = {
            new_status: f"{event_summary} (第{current_chapter}章)"
        }

        # 更新角色A和角色B的关系字典
        kb["characters"][character_a]["relationship"][character_b] = relationship_entry
        kb["characters"][character_b]["relationship"][character_a] = relationship_entry

        save_knowledge_base(kb, knowledge_path)
        return f"✅ 关系更新：{character_a} ↔ {character_b}（{new_status}）"
    else:
        # 如果关系没有变化，则不执行任何操作
        return f"ℹ️  关系未变：{character_a} 与 {character_b} 已是 '{new_status}'，跳过更新"


def update_character_backstory(name: str, new_information: str,
                               mode: Literal['append', 'overwrite'] = 'append',
                               current_chapter: Optional[int] = None) -> str:
    """更新角色个人背景故事

    Args:
        name: 角色姓名
        new_information: 新的背景信息
        mode: 更新模式 - 'append'追加, 'overwrite'覆盖
        current_chapter: 当前章节（可选，用于添加章节标记）
    """
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")

    dir5 = "knowledge_base.json"

    knowledge_path = os.path.join(dir1, dir2, dir5)
    kb = get_knowledge_base(knowledge_path)

    if name not in kb["characters"]:
        return f"❌ 错误：角色 '{name}' 不存在，无法更新背景故事"

    # 添加章节标记（如果提供了当前章节）
    if current_chapter is not None:
        formatted_info = f"第{current_chapter}章：{new_information}"
    else:
        formatted_info = new_information

    character = kb["characters"][name]

    if mode == 'append':
        if character["backstory"]:
            character["backstory"] += f"\n{formatted_info}"
        else:
            character["backstory"] = formatted_info
    elif mode == 'overwrite':
        character["backstory"] = formatted_info

    save_knowledge_base(kb, knowledge_path)
    return f"✅ 成功更新角色 '{name}' 的个人背景故事"


def analyze_relationship_changes(character_interactions: List[Dict],
                                 chapter_text: str, current_chapter: int) -> List[Dict]:
    """第三步：分析关系变化 - 使用输出解析器版本"""
    tool_calls = []
    if not character_interactions:
        print("ℹ️  未识别到角色互动，无需更新关系")
        return tool_calls

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=EvidenceText)

    prompt_template = """
任务：从以下章节内容中，提取角色'{char_a}'和'{char_b}'互动的具体文本片段。
要求：仅返回提取的文本（最多300字符），不要添加任何解释。

章节内容：
{chapter_text}

{format_instructions}
"""

    master_llm = get_evaluation_llm()

    for interaction in character_interactions:
        char_a = interaction.get("character_a", "").strip()
        char_b = interaction.get("character_b", "").strip()
        interaction_type = interaction.get("interaction_type", "陌生人").strip()
        summary = interaction.get("interaction_summary", "无描述").strip()

        if not char_a or not char_b:
            continue

        try:
            # 使用输出解析器提取证据
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["char_a", "char_b", "chapter_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt | master_llm
            raw_response = chain.invoke({
                "char_a": char_a,
                "char_b": char_b,
                "chapter_text": chapter_text[:1000]  # 限制文本长度
            })

            # 过滤思考标签并解析
            from helper import filter_think_tags, parse_json_with_filtering
            filtered_content = filter_think_tags(raw_response.content)
            parsed_evidence = parse_json_with_filtering(filtered_content, parser)

            evidence = parsed_evidence.evidence.strip() or "未提取到具体文本"

        except Exception as e:
            print(f"--- ❌ 证据提取失败: {e} ---")
            evidence = "未提取到具体文本"

        tool_calls.append({
            "name": "update_relationship",
            "args": {
                "character_a": char_a,
                "character_b": char_b,
                "new_status": interaction_type,
                "event_summary": summary,
                "chapter_evidence": evidence,
                "current_chapter": current_chapter,
            }
        })

    print(f"✅ 生成关系更新任务：{len(tool_calls)}个")
    return tool_calls


def detect_personal_backstory_updates(characters: List[Dict], chapter_text: str, current_chapter: int) -> List[Dict]:
    """检测并生成个人背景故事更新任务 - 使用输出解析器版本"""
    tool_calls = []

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=BackstorySummary)

    prompt_template = """
任务：分析以下章节内容，提取角色'{name}'的个人背景故事信息。
提取与角色个人相关，比如：
- 角色的个人经历、回忆
- 角色的内心独白、想法
- 角色的技能、特长、习惯
- 角色的个人目标、梦想
同时可以少部分包含其他角色的影响,比如：
其他角色使得该角色发生了变化等

章节内容：
{chapter_text}

要求：返回一个简短的总结（不超过100字），如果没有相关的个人背景信息，返回"无"

{format_instructions}
"""

    master_llm = get_evaluation_llm()

    for char in characters:
        name = char.get("name", "").strip()
        if not name:
            continue

        try:
            # 使用输出解析器提取背景故事
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["name", "chapter_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt | master_llm
            raw_response = chain.invoke({
                "name": name,
                "chapter_text": chapter_text[:1500]  # 限制文本长度
            })

            # 过滤思考标签并解析
            from helper import filter_think_tags, parse_json_with_filtering
            filtered_content = filter_think_tags(raw_response.content)
            parsed_summary = parse_json_with_filtering(filtered_content, parser)

            backstory_summary = parsed_summary.summary.strip()

        except Exception as e:
            print(f"--- ❌ 背景故事提取失败: {e} ---")
            backstory_summary = "无"

        if backstory_summary and backstory_summary != "无":
            tool_calls.append({
                "name": "update_character_backstory",
                "args": {
                    "name": name,
                    "new_information": backstory_summary,
                    "mode": "append",
                    "current_chapter": current_chapter
                }
            })

    return tool_calls

def run_complete_relationship_analysis(chapter_text: str, current_chapter: int = 1):
    """完整的角色和关系分析流程（已添加个人背景故事更新）"""
    print(f"\n" + "=" * 50)
    print(f"🔍 开始分析第{current_chapter}章")
    print("=" * 50)

    # 第一步：识别角色
    print("\n📝 第一步：识别章节中的角色...")
    identification_result = identify_characters_in_text(chapter_text, current_chapter)
    characters = identification_result.get("characters", [])
    interactions = identification_result.get("character_interactions", [])

    # 第二步：创建/更新角色节点
    print("\n👤 第二步：处理角色节点（创建/更新）...")
    character_tool_calls = []
    for char in characters:
        name = char.get("name", "").strip()
        if not name:
            continue
        character_tool_calls.append({
            "name": "create_character_if_not_exists",
            "args": {
                "name": name,
                "age": char.get("estimated_age"),
                "gender": char.get("gender"),
                "appearance": (char.get("appearance") or "").strip(),
                "traits": [str(t).strip() for t in char.get("traits", []) if str(t).strip()],
                "specialties": [str(t).strip() for t in char.get("specialties", []) if str(t).strip()],
                "hobbies": [str(t).strip() for t in char.get("hobbies", []) if str(t).strip()],
                "occupations": [str(t).strip() for t in char.get("occupations", []) if str(t).strip()],
                "current_chapter": current_chapter,
                "relationship": {}
            }
        })
    print(f"📋 生成角色操作任务：{len(character_tool_calls)}个")

    # 第三步：检测个人背景故事更新
    print("\n📖 第三步：检测个人背景故事更新...")
    backstory_tool_calls = detect_personal_backstory_updates(characters, chapter_text, current_chapter)
    print(f"📋 生成背景故事更新任务：{len(backstory_tool_calls)}个")

    # 第四步：分析关系变化
    print("\n🔗 第四步：分析角色关系变化...")
    relationship_tool_calls = analyze_relationship_changes(
        interactions,
        chapter_text,
        current_chapter
    )
    print(f"📋 生成关系更新任务：{len(relationship_tool_calls)}个")

    all_tool_calls = character_tool_calls + backstory_tool_calls + relationship_tool_calls
    print(f"\n✅ 分析完成：共生成 {len(all_tool_calls)} 个操作任务")
    return all_tool_calls


def execute_tool_calls(tool_calls: List[Dict]):
    """执行工具调用"""
    available_tools = {
        "create_character_if_not_exists": create_character_if_not_exists,
        "update_character_backstory": update_character_backstory,  # 添加这个
        "update_relationship": update_relationship
    }

    results = []
    if not tool_calls:
        return results

    print(f"\n🛠️  开始执行 {len(tool_calls)} 个工具调用...")
    for idx, call in enumerate(tool_calls, 1):
        print(f"\n--- 任务{idx}/{len(tool_calls)}：{call['name']} ---")
        if call["name"] not in available_tools:
            err_msg = f"❌ 未知工具：{call['name']}"
            print(err_msg)
            results.append(err_msg)
            continue
        try:
            tool_func = available_tools[call["name"]]
            result = tool_func(**call["args"])
            print(f"✅ 结果：{result}")
            results.append(result)
        except Exception as e:
            err_msg = f"❌ 执行失败：{str(e)}"
            print(err_msg)
            results.append(err_msg)
    return results


def simulate_user_confirmation_and_execute(tool_calls: List[Dict]):
    """模拟用户确认并执行"""
    print("\n" + "=" * 60)
    print(f"👤 请确认以下知识库更新操作（共{len(tool_calls)}个）")
    print("=" * 60)

    if not tool_calls:
        print("ℹ️  无待执行操作，无需确认")
        return []

    char_ops = [c for c in tool_calls if c["name"] == "create_character_if_not_exists"]
    rel_ops = [c for c in tool_calls if c["name"] == "update_relationship"]

    if char_ops:
        print("\n📝 【角色操作】")
        for op in char_ops:
            args = op["args"]
            print(f"  • {args['name']}")
            details = []
            if args.get("age"): details.append(f"年龄：{args['age']}")
            if args.get("gender"): details.append(f"性别：{args['gender']}")
            if args.get("traits"): details.append(f"性格：{', '.join(args['traits'])}")
            if details: print(f"    （{', '.join(details)}）")

    if rel_ops:
        print("\n🔗 【关系操作】")
        for op in rel_ops:
            args = op["args"]
            print(f"  • {args['character_a']} ↔ {args['character_b']}")
            print(f"    关系类型：{args['new_status']}")
            print(f"    事件摘要：{args['event_summary'][:50]}...")
    return execute_tool_calls(tool_calls)


# ================================================================= #
# 5. 测试函数
# ================================================================= #

def character_analysis():
    """测试角色分析功能"""
    print("🧪 测试角色分析输出解析...")

    test_text = """
    林屿站在窗前，看着外面的雨景。他的室友苏晓棠正在客厅画画。
    "你今天看起来心情不错，"苏晓棠说道，手中的画笔不停。
    林屿转过身，微微一笑："是啊，今天收到了一家出版社的回复。"
    """

    try:
        result = identify_characters_in_text(test_text, 1)
        print("✅ 角色分析测试通过")
        print(f"识别到 {len(result['characters'])} 个角色")
        for char in result['characters']:
            print(f"  - {char['name']}")
    except Exception as e:
        print(f"❌ 角色分析测试失败: {e}")


if __name__ == "__main__":
    character_analysis()