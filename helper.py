import json
import os
import re
from typing import Dict, Any, Optional, List, Tuple

import dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from memory import MemorySystem
from state import StoryState
from utils import get_llm, get_embedding_llm, get_evaluation_llm


# ==================== 定义 Pydantic 模型 ====================

class ChapterPlan(BaseModel):
    """章节计划模型"""
    chapter_title: str = Field(description="引人入胜的章节标题")
    chapter_outline: str = Field(description="详细的、包含主要情节点的章节大纲")
    creative_brief: Dict[str, List[str]] = Field(description="创作指令")


class StorySuggestion(BaseModel):
    """故事建议模型"""
    title: str = Field(description="章节标题")
    story_outline: str = Field(description="整个故事的发展过程")
    total_text_style: str = Field(description="写作风格、语气和重点表现手法")


class ChapterSummary(BaseModel):
    """章节摘要模型"""
    summary: str = Field(description="章节内容的简洁摘要")


def filter_think_tags(text: str) -> str:
    """
    过滤掉文本中 <think> 和 </think> 标签及其之间的内容
    """
    if not text:
        return text

    # 使用正则表达式移除 <think>...</think> 标签及其内容
    filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 移除可能残留的 <think> 或 </think> 标签（不完整的情况）
    filtered_text = re.sub(r'<think>|</think>', '', filtered_text)

    # 移除多余的空行和空白字符
    filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text).strip()

    return filtered_text


def parse_json_with_filtering(text: str, parser) -> Any:
    """
    先过滤掉思考标签，然后尝试解析JSON
    """
    try:
        # 首先尝试直接解析
        return parser.parse(text)
    except Exception as e:
        # 如果直接解析失败，尝试过滤思考标签后再解析
        filtered_text = filter_think_tags(text)
        try:
            return parser.parse(filtered_text)
        except Exception as e2:
            # 如果还是失败，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', filtered_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return parser.parse(json_str)
                except Exception as e3:
                    raise e3
            raise e2


def get_best_draft(state: StoryState) -> Dict[str, Any]:
    print("--- 📖 Get Best Chapter: Selecting best version.. ---")
    # --- 关键修改：将最后一个版本也加入候选列表 ---
    current_versions = state.get('chapter_versions', [])
    last_decision = state.get('committee_decision')
    last_draft = state.get('revised_draft')

    if last_decision and last_draft:
        scores = last_decision.get('dimension_scores', {})
        numeric_scores = [s['score'] for s in scores.values() if isinstance(s.get('score'), (int, float))]
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
        current_versions.append({
            "draft": last_draft,
            "scores": scores,
            "average_score": avg_score
        })

    if not current_versions:
        print("--- ❌ Error: No chapter versions found to publish. ---")
        # 回退到使用最后的草稿
        best_draft = last_draft or "Error: No content available."
    else:
        # --- 关键修改：找到分数最高的版本 ---
        best_version = max(current_versions, key=lambda x: x['average_score'])
        best_draft = best_version['draft']
        best_score = best_version['average_score']
        print(f"--- 🏆 Best version selected with average score: {best_score:.2f} ---")
    return {"final_chapter": best_draft}


def publish_chapter(state: StoryState, best_draft: str) -> StoryState:
    """
    发布章节 Agent - v2.0
    - v2.0: 从所有修订版本中选择平均分最高的进行发布。
    """
    print("--- 📖 Publish Chapter: Selecting best version and publishing to memory... ---")

    chapter_index = state['current_chapter_index']
    chapter_title = state['chapter_title']

    # 使用输出解析器生成摘要
    print("--- 📝 Generating chapter summary for the best version... ---")
    summary = generate_chapter_summary(chapter_title, best_draft)

    memory_dir = get_memory_path()
    memory = MemorySystem.load(memory_dir, get_embedding_llm())

    # 将最佳章节添加到记忆系统
    memory.add_chapter(
        chapter_index=chapter_index,
        chapter_title=chapter_title,
        full_text=best_draft,
        summary=summary
    )

    # 更新历史记录
    full_text_history = state.get('full_text_history', []) + [best_draft]
    summary_history = state.get('summary_history', []) + [summary]

    print(f"--- ✅ Successfully published Chapter {chapter_index}: '{chapter_title}' ---")
    print(f"Summary: {summary[:100]}...")
    state["published_chapter"] = best_draft
    state["full_text_history"] = full_text_history
    state["summary_history"] = summary_history
    current_chapter_index = state["current_chapter_index"]
    state["current_chapter_index"] = current_chapter_index + 1

    return state


def generate_chapter_summary(chapter_title: str, chapter_content: str) -> str:
    """使用输出解析器生成章节摘要"""
    parser = PydanticOutputParser(pydantic_object=ChapterSummary)

    prompt_template = """
请为以下章节内容生成一个简洁的摘要（不超过150字）：

**章节标题**: {chapter_title}
**章节内容**: {chapter_content}

{format_instructions}
"""

    llm = get_evaluation_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["chapter_title", "chapter_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({
            "chapter_title": chapter_title,
            "chapter_content": chapter_content
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_summary = parse_json_with_filtering(filtered_content, parser)

        return parsed_summary.summary

    except Exception as e:
        print(f"--- ❌ Error generating summary: {e} ---")
        # 回退到简单的截断摘要
        return chapter_content[:147] + "..." if len(chapter_content) > 150 else chapter_content


def update_json_with_dict(file_path, updates):
    """
    根据提供的字典更新 JSON 文件中的字段信息。
    只更新字典中存在于 JSON 文件中的键，跳过不存在的键。
    支持嵌套字典和列表的更新。

    Args:
        file_path (str): JSON 文件的路径
        updates (dict): 包含要更新的字段及其新值的字典
    """
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 加载 JSON 数据

        # 定义一个递归函数来更新嵌套字典


        def update_nested_dict(target_dict, updates_dict):
            for key, value in updates_dict.items():
                if key in target_dict:
                    # 如果值是字典，则递归更新
                    if isinstance(target_dict[key], dict) and isinstance(value, dict):
                        update_nested_dict(target_dict[key], value)
                    # 如果值是列表，则更新整个列表
                    elif isinstance(target_dict[key], list) and isinstance(value, list):
                        target_dict[key] = value
                    # 否则直接更新值
                    else:
                        target_dict[key] = value
                else:
                    print(f"跳过更新：键 '{key}' 不存在于 JSON 文件中。")

        # 更新字段值
        update_nested_dict(data, updates)

        # 写回 JSON 文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)  # 格式化写入

        print("JSON 文件已成功更新存在的键！")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在！")
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是一个有效的 JSON 文件！")
    except Exception as e:
        print(f"发生错误：{e}")


def initial_memory(json_path: str, memory_path: str):
    embeddings = get_embedding_llm()
    memory = MemorySystem.load(memory_path, embedding_model=embeddings)
    with open(json_path, "r", encoding="utf-8") as f:
        story_data = json.load(f)
    story_data["memory"] = memory


def get_memory_path():
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")
    memory_path = os.path.join(dir1, dir2)
    return memory_path


# (确保这在 planned 函数所在的单元格)
def planned(user_input: str, json_path: str, memory_path: str, memory_system: Optional[MemorySystem] = None):
    """
    章节规划核心引擎函数 V2.1。
    - 接受一个可选的 memory_system 实例以避免重复加载。
    """
    STORY_JSON_FILE = json_path
    MEMORY_DIR = memory_path
    print(json_path)
    print(memory_path)
    print("\n" + "=" * 50)
    print("🚀 Executing Chapter Planning Function...")
    print(f"User Input: \"{user_input}\"")
    print("=" * 50)

    try:
        # 加载 story_data
        with open(STORY_JSON_FILE, "r", encoding="utf-8") as f:
            story_data = json.load(f)

        title = story_data.get("title", "未命名故事")
        story_outline = story_data.get("story_outline", "无大纲")
        current_chapter_index = story_data.get("current_chapter_index", 0)
        chapter_data = {
            "chapter_title": story_data.get("chapter_title"),
            "chapter_outline": story_data.get("chapter_outline"),
            "creative_brief": story_data.get("creative_brief"),
        }
        # 上一章内容回顾
        full_text_history = story_data["full_text_history"]
        previous_chapter_section = "### 上一章内容回顾\n(这是故事的第一章，没有前文。)"
        if full_text_history:
            previous_chapter_section = f"""### 上一章内容回顾 (请确保你的创作与之无缝衔接)
        {full_text_history[-1]}
        """
        summary_history = story_data["summary_history"]
        # 新增：上上上章的总结回顾
        three_chapters_back_summary = ""
        if len(summary_history) >= 3:
            three_chapters_back_summary = f"""### 前三章总结回顾 (提供更久远的故事背景)
        {summary_history[-3]}
        """
        chapter_json = json.dumps(chapter_data, indent=4, ensure_ascii=False)

        # --- 【关键修改点】 ---
        # 如果没有传入 memory_system 实例，则加载它
        if memory_system is None:
            print("--- No MemorySystem instance provided, loading from disk... ---")
            embeddings = get_embedding_llm()
            memory = MemorySystem.load(MEMORY_DIR, embedding_model=embeddings)
        else:
            print("--- Using provided MemorySystem instance. ---")
            memory = memory_system

        # ... (后续的检索和规划逻辑保持不变) ...
        print("\n2. Preparing and executing vector database retrieval...")
        retrieval_query = user_input
        retrieved_context = memory.retrieve_context_for_writer(query=retrieval_query)
        print("✅ Retrieval complete.")

        print("\n3. Constructing prompt and calling LLM for chapter planning...")

        # 使用输出解析器
        parser = PydanticOutputParser(pydantic_object=ChapterPlan)

        prompt_template = """
你是一位顶级的小说主编，负责为故事的当前章节制定详细的创作计划。

### 故事背景
- **故事标题**: {title}
- **故事总大纲**: {story_outline}
- **上一章节内容**: {previous_chapter_section}
- **前三章节的摘要**: {three_chapters_back_summary}
### 上下文与指令
1. **从记忆库检索到的相关历史情节**:\n{retrieved_context}
2. **现在已有的创作指令**:\n{chapter_json}
3. **用户的最新指令 (最重要)**:\n\"{user_input}\"

### 你的任务
请综合以上所有信息，特别是**用户的最新指令**，为当前章节（第 {current_chapter_index} 章）生成一份最终的、优化过的创作计划。
你生成的chapter_outline应该在300字左右

{format_instructions}
"""

        llm = get_llm()

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["title", "story_outline", "retrieved_context", "chapter_json",
                             "user_input", "current_chapter_index", "previous_chapter_section", "three_chapters_back_summary"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({
            "title": title,
            "story_outline": story_outline,
            "retrieved_context": retrieved_context,
            "chapter_json": chapter_json,
            "user_input": user_input,
            "current_chapter_index": current_chapter_index,
            "previous_chapter_section": previous_chapter_section,
            "three_chapters_back_summary": three_chapters_back_summary
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        new_plan = parse_json_with_filtering(filtered_content, parser)

        # 更新 JSON 文件

        story_data["chapter_title"] = new_plan.chapter_title
        story_data["chapter_outline"] = new_plan.chapter_outline
        story_data["creative_brief"] = new_plan.creative_brief
        story_data["committee_decision"] = None

        with open(STORY_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(story_data, f, ensure_ascii=False, indent=4)

        print(f"✅ '{STORY_JSON_FILE}' updated successfully for Chapter {story_data['current_chapter_index']}.")

    except Exception as e:
        print(f"❌ 在规划函数中发生未知错误: {e}")


def put_chapter(chapter_title: str, chapter_content: str, chapter_path: str):
    """
        保存章节内容到JSON文件中

        参数:
            chapter_title (str): 章节标题
            chapter_content (str): 章节内容
            chapter_path (str): JSON文件路径
        """
    # 检查文件是否存在
    if os.path.exists(chapter_path):
        with open(chapter_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 文件不存在，创建并初始化
        data = {'current_index': 1}
    key = f"第{data['current_index']}章 {chapter_title}"
    data[key] = chapter_content

    # current_index加1
    data['current_index'] += 1

    # 保存回文件
    with open(chapter_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_all_chapter(chapter_path: str) -> Dict[str, str]:
    with open(chapter_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def finalize_chapter_and_save_state(final_state: StoryState, json_path: str, memory_path: str):
    """
    工作流后处理函数。
    - 将工作流的最终状态进行清理和整合。
    - 保存更新后的MemorySystem。
    - 将持久化的故事状态写回JSON文件，为下一次规划做准备。
    """
    print("\n" + "=" * 50)
    print("🚀 Executing Post-Workflow Finalization...")
    print(f"Updating state in '{json_path}'")
    print("=" * 50)

    try:
        # 1. 验证最终状态是否有效
        if not final_state or not final_state.get("published_chapter"):
            print("❌ 错误: 最终状态无效或章节未发布，无法进行最终处理。")
            return

        memory = MemorySystem.load(memory_path, get_embedding_llm())
        # 3. 准备要持久化到JSON的新状态
        # 这些是需要跨章节保留的关键信息
        persistent_story_data = {
            "title": final_state.get("title"),
            "story_outline": final_state.get("story_outline"),
            "current_chapter_index": final_state.get("current_chapter_index"),
            "full_text_history": final_state.get("full_text_history"),
            "summary_history": final_state.get("summary_history"),
            "committee_decision": final_state.get("committee_decision"),  # 保存这次的决策，为下一次规划提供参考

            # --- 重置所有临时字段，为下一章做准备 ---
            "chapter_title": "",
            "chapter_outline": "",
            "creative_brief": {
                "narrative_goals": [
                ],
                "character_focus": [
                ],
                "thematic_elements": [
                ],
                "structural_requirements": [
                ]
            },
            "initial_draft": None,
            "revised_draft": None,
            "rewrite_attempts": 0,
            "suggestions": [],
            "expert_evaluations": [],
            "published_chapter": None,
            "agent_flags": {},
            "required_agents": [],
            "chapter_versions": [],
            "revision_brief": None,
            "final_chapter": [],
        }
        print("✅ Prepared persistent state for the next chapter.")

        # 4. 将新状态写入JSON文件
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(persistent_story_data, f, ensure_ascii=False, indent=4)

        print(f"✅ Successfully updated '{json_path}'. The system is ready for the next chapter planning.")
        print("=" * 50)
        print("🎉 Finalization complete.")
        print("=" * 50)

    except KeyError as e:
        print(f"❌ 错误: 最终状态中缺少必要的键: {e}")
    except Exception as e:
        print(f"❌ 在后处理函数中发生未知错误: {e}")
        import traceback
        traceback.print_exc()


def _parse_llm_response(response: str) -> Dict[str, str]:
    """使用正则表达式解析LLM的结构化输出，以提高健壮性。"""
    response = response.strip()
    parsed_suggestions = {"title": "", "story_outline": "", "total_text_style": ""}

    try:
        title_match = re.search(r"title:\s*(.*?)\s*story_outline:", response, re.DOTALL)
        if title_match:
            parsed_suggestions["title"] = title_match.group(1).strip()

        outline_match = re.search(r"story_outline:\s*(.*?)\s*total_text_style:", response, re.DOTALL)
        if outline_match:
            parsed_suggestions["story_outline"] = outline_match.group(1).strip()

        style_match = re.search(r"total_text_style:\s*(.*)", response, re.DOTALL)
        if style_match:
            parsed_suggestions["total_text_style"] = style_match.group(1).strip()
    except Exception as e:
        print(f"[错误] 解析LLM响应时出错: {e}")
        # 在解析失败时，将原始响应放入其中一个字段，以便调试
        parsed_suggestions["raw_response"] = response

    return parsed_suggestions


def get_ai_suggestions(user_input: str) -> Dict[str, str]:
    """
    根据用户输入，生成AI建议的小说标题、大纲和风格指导。

    Args:
        user_input: 用户提供的创作需求。

    Returns:
        包含建议字典的元组。
    """
    print(f"\n--- 收到用户需求: \"{user_input}\" ---")
    print("--- 🤖 正在生成AI创作建议... ---")

    # 使用输出解析器
    parser = PydanticOutputParser(pydantic_object=StorySuggestion)

    prompt_template = """
你是一个小说生成助手，根据用户输入生成AI建议的标题、大纲和风格指导。

请根据用户的创作需求，生成小说章节的建议标题、详细大纲和风格指导。

用户需求：{user_input}

{format_instructions}
"""

    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["user_input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({"user_input": user_input})

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_suggestion = parse_json_with_filtering(filtered_content, parser)

        return parsed_suggestion.model_dump()

    except Exception as e:
        print(f"--- ❌ Error generating AI suggestions: {e} ---")
        # 回退到原始方法
        return get_ai_suggestions_fallback(user_input)


def get_ai_suggestions_fallback(user_input: str) -> Dict[str, str]:
    """回退方法：使用原始的JSON解析方式"""
    prompt_template = f"""
你是一个小说生成助手，根据用户输入生成AI建议的标题、大纲和风格指导。

请根据用户的创作需求，生成小说章节的建议标题、详细大纲和风格指导，并以字典格式返回。

用户需求：{user_input}

输出格式要求：
- 字典中必须包含以下键值对：
  - 'title': 提供1个最符合需求的章节标题（字符串类型）
  - 'story_outline': 使用一段话介绍整个故事的发展过程（字符串类型）
  - 'total_text_style': 描述适合该章节的写作风格、语气和重点表现手法（字符串类型）

示例：
{{
  "title": "初遇的咖啡馆",
  "story_outline": "故事发生在一个温馨的咖啡馆，女主角林雨是一个热爱咖啡与书籍的年轻作家，因一次偶然的机会，在这里邂逅了男主角赵晨，一个刚刚回国的摄影师。两人在咖啡馆的第一次相遇充满了火花，林雨被赵晨的幽默与才华吸引，而赵晨也对林雨的独立与热情印象深刻。随着时间的推移，两人开始频繁见面，分享彼此的梦想与生活。在这个过程中，他们不仅彼此扶持，在各自的事业上也取得了进展。然而，随着赵晨事业的发展，面临的选择使得两人的关系变得紧张，林雨需要面对自己的情感与梦想的抉择，最终两人能否在事业与爱情之间找到平衡，成为了故事的核心冲突。",
  "total_text_style": "写作风格：以细腻而富有诗意的语言描绘都市生活的美好与复杂，抒情而不失真实。使用丰富的意象，如咖啡的香气、书页的翻动等，来传达人物内心的情感波动。语气温暖而亲切，时而带有淡淡的忧伤，强调人物之间的情感交流与内心挣扎。叙述中融入细腻的心理描写，展现人物的成长与变化，同时通过对话与日常细节来推动情节发展，保持故事的流畅性与真实感。故事聚焦在爱情与梦想的交织中，让读者在平凡的生活中感受到爱的力量与希望的光芒。"
}}

请直接返回字典格式内容，不要添加多余的文字或解释。
"""

    llm = get_llm()
    response = llm.invoke(prompt_template).content

    try:
        story_dict = json.loads(response)
        return story_dict
    except json.JSONDecodeError:
        print("--- ❌ Failed to parse JSON, using regex fallback ---")
        return _parse_llm_response(response)


def route(state: StoryState) -> StoryState:  # 返回类型应该是 StoryState
    # 直接在传入的 state 对象上创建 agent_flags 键
    state["agent_flags"] = {
        "emotional_reader_agent": 0,
        "rhythm_reader_agent": 0,
        "immersion_reader_agent": 0,
        "structural_novelist_agent": 0,
        "foreshadowing_novelist_agent": 0
    }
    if state['revised_draft'] is not None:
        state['initial_draft'] = state['revised_draft']
        state['revised_draft'] = None
    # 根据 required_agents 更新 flags
    for k in state["required_agents"]:
        state["agent_flags"][k] = 1

    # 返回被修改后的 state 对象
    return state


def decide_to_publish_or_rewrite(state: StoryState) -> str:
    """
    仅根据需要重写的Agent列表和重写次数决定下一步操作。
    """
    print("--- 🔍 决策中：发布或重写... ---")
    rewrite_attempts = state.get('rewrite_attempts', 0)
    required_agents = state.get("required_agents", [])

    # 如果需要重写的agent列表不为空，并且重写次数小于3次，则选择重写
    if required_agents and rewrite_attempts < 1:
        print(f"--- 裁决：需要重写 (当前尝试次数: {rewrite_attempts}) ---")
        return "rewrite_chapter"
    else:
        # 否则，直接发布（包括重写超过3次被强制发布的情况）
        print("--- 裁决：发布章节 ---")
        return "publish_chapter"


def prepare_for_rewrite(state: StoryState) -> dict:
    """
    在每次重写循环开始前，更新状态。
    - v2.0: 保存当前草稿及其评分，并为下一轮建议者提供针对性简报。
    """
    rewrite_attempts = state.get('rewrite_attempts', 0) + 1

    # --- 关键修改：保存当前版本及其评分 ---
    current_versions = state.get('chapter_versions', [])
    last_decision = state.get('committee_decision')
    last_draft = state.get('revised_draft')

    revision_brief_for_next_loop = {}
    if last_decision and last_draft:
        scores = last_decision.get('dimension_scores', {})
        # 计算平均分
        numeric_scores = [s['score'] for s in scores.values() if isinstance(s.get('score'), (int, float))]
        avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0

        # 将此版本存入历史记录
        current_versions.append({
            "draft": last_draft,
            "scores": scores,
            "average_score": avg_score
        })
        print(f"--- 💾 保存修订版本 (第 {rewrite_attempts - 1} 次修订), 平均分: {avg_score:.2f} ---")

        # 为下一次循环准备针对性简报
        revision_brief_for_next_loop = scores

    print("==========================")
    print(f"       重写次数 {rewrite_attempts}       ")
    print("==========================")

    # 返回一个字典来更新状态
    return {
        "rewrite_attempts": rewrite_attempts,
        "chapter_versions": current_versions,
        "revision_brief": revision_brief_for_next_loop,  # 注入针对性简报
        "initial_draft": state["revised_draft"],  # 将已修订的草稿作为下一轮的初稿
        "revised_draft": None,
        "suggestions": [],
        "expert_evaluations": [],
        "committee_decision": None
    }


# ==================== 测试函数 ====================

def output_parsers():
    """测试所有输出解析器功能"""
    print("🧪 测试输出解析器功能...")

    # 测试故事建议
    try:
        suggestions = get_ai_suggestions("一个关于太空探险的故事")
        print("✅ AI建议测试通过")
        print(f"标题: {suggestions.get('title')}")
    except Exception as e:
        print(f"❌ AI建议测试失败: {e}")

    # 测试摘要生成
    try:
        summary = generate_chapter_summary("测试章节", "这是一个测试章节的内容")
        print("✅ 摘要生成测试通过")
        print(f"摘要: {summary}")
    except Exception as e:
        print(f"❌ 摘要生成测试失败: {e}")

    print("🎉 输出解析器测试完成")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 运行测试
    output_parsers()