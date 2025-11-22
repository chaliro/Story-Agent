from typing import Dict
from pydantic import BaseModel, Field

from character import get_knowledge_base
import os
import dotenv

from helper import get_memory_path, filter_think_tags, parse_json_with_filtering
from state import StoryState
from utils import get_llm, get_embedding_llm
from memory import MemorySystem
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()


# ==================== 定义 Pydantic 模型 ====================

class WriterOutput(BaseModel):
    initial_draft: str = Field(description="生成的章节草稿内容")



class RewriterOutput(BaseModel):
    revised_draft: str = Field(description="修订后的章节草稿内容")

class DistillOutput(BaseModel):
    distilled_brief: list[str] = Field(description="修订后的章节草稿内容")

from character import get_knowledge_base, get_story_graph_context

def writer_agent(state: StoryState) -> Dict[str, str]:
    """作家Agent - 生成章节初稿"""
    print(f"--- 📝 Writer Agent: Drafting Chapter {state['current_chapter_index']}: '{state['chapter_title']}' ---")

    chapter_title = state['chapter_title']
    chapter_outline = state['chapter_outline']
    creative_brief = state['creative_brief']
    full_text_history = state.get('full_text_history', [])
    summary_history = state.get('summary_history', [])  # 新增：获取摘要历史
    style_guide = state.get("total_text_style", "无特定风格要求")

    # 新增：获取核心叙事目标
    narrative_goals = creative_brief.get("narrative_goals", [])

    # 新增：从知识库获取与叙事目标相关的角色信息
    character_relevant_info = ""
    kb = get_knowledge_base(os.getenv("KNOWLEDGE_BASE_PATH"))
    if narrative_goals and kb.get("characters"):
        character_relevant_info = "### 与核心叙事目标相关的角色信息\n"
        # 重点关注的角色
        character_focus = creative_brief.get("character_focus", [])

        for char_name, char_data in kb["characters"].items():
            # 如果是重点关注的角色，或者角色特征与叙事目标相关
            if char_name in character_focus or any(
                    goal.lower() in ' '.join(char_data["traits"]).lower() for goal in narrative_goals):
                character_relevant_info += f"- {char_name}：\n"
                character_relevant_info += f"  身份：{char_data.get('identity', '未知')}\n"
                character_relevant_info += f"  性格特点：{', '.join(char_data.get('traits', []))}\n"
                character_relevant_info += f"  职业：{', '.join(char_data.get('occupations', []))}\n"
                character_relevant_info += f"  特长：{', '.join(char_data.get('specialties', []))}\n"
                character_relevant_info += f"  爱好：{', '.join(char_data.get('hobbies', []))}\n"
                character_relevant_info += f"  首次出现：第{char_data.get('first_appearance_chapter', '未知')}章\n"

                # 添加与其他重点关注角色的关系
                relationships = char_data.get("relationship", {})
                related_chars = []
                for other_char, rel_details in relationships.items():
                    # 只保留与重点关注角色的关系
                    if other_char in character_focus:
                        # 提取关系类型（如"合租室友"）和描述
                        rel_type = next(iter(rel_details.keys())) if rel_details else "未知关系"
                        rel_desc = rel_details.get(rel_type, "")
                        related_chars.append(f"{other_char}（{rel_type}）：{rel_desc}")

                if related_chars:
                    character_relevant_info += "  与重点角色关系：\n"
                    for rel in related_chars:
                        character_relevant_info += f"    - {rel}\n"
                else:
                    character_relevant_info += "  与重点角色关系：无\n"

                character_relevant_info += "\n"

    memory_dir = get_memory_path()
    memory = MemorySystem.load(memory_dir, get_embedding_llm())


    # 上一章内容回顾
    previous_chapter_section = "### 上一章内容回顾\n(这是故事的第一章，没有前文。)"
    if full_text_history:
        previous_chapter_section = f"""### 上一章内容回顾 (请确保你的创作与之无缝衔接)
{full_text_history[-1]}
"""

    # 新增：上上上章的总结回顾
    three_chapters_back_summary = ""
    if len(summary_history) >= 3:
        three_chapters_back_summary = f"""### 前三章总结回顾 (提供更久远的故事背景)
{summary_history[-3]}
"""

    # 长期历史背景回顾 - 只检索前三个summary
    retrieval_query = f"章节大纲: {chapter_outline}\n创作指令: {', '.join(f'{k}: {v}' for k, v in creative_brief.items())}"
    long_term_context = memory.retrieve_context_for_writer(query=retrieval_query)

    # 只保留前三个摘要
    if long_term_context:
        # 假设返回的是按相关性排序的摘要列表，这里简单处理为按换行分割取前三个
        context_items = long_term_context.split('\n\n')[:3]
        long_term_context = '\n\n'.join(context_items)

    # 风格指南
    style_instruction_section = f"""
### 故事总体风格指南 (必须严格遵守)
你的所有创作都必须遵循以下风格：
**{style_guide}**

#### 如何应用风格的示例:
这部分是教你如何应用上述风格，请理解其精髓，不要模仿具体内容。

**[示例1]**
* **指定的风格要求**: "写作风格：以情节和主角内心活动为绝对核心，快节奏推进。环境描写仅在必要时点到为止，不做过多渲染。"
* **不佳的写法 (❌ 违反风格)**: "月光透过哥特式窗户的彩色玻璃，在地上投下斑驳的影子。空气中弥漫着旧书和灰尘的味道，古老的书架上布满了蜘蛛网，一直延伸到高高的拱形天花板。房间中央的橡木桌上，雕刻着繁复的藤蔓花纹，显得庄严肃穆。" (这段描写过于关注环境，拖慢了节奏)。
* **优秀的写法 (✅ 符合风格)**: "她推开门，心猛地一沉。这间书房就是她要找的地方。她迅速扫了一眼布满灰尘的书架，目光立刻锁定在房间中央那张巨大的橡木桌上——日记里提到的线索一定就在那里。她一边警惕着门口的动静，一边快步走了过去，每一步都感觉离危险更近。" (这段描写聚焦于主角的目标、内心感受和行动，节奏很快)。
"""

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=WriterOutput)
    # 获取本章关注的角色
    character_focus = creative_brief.get("character_focus", [])

    # === 🔴 核心修改点：调用图谱推理引擎 ===
    print(f"--- 🕸️ GraphRAG: Reasoning for characters {character_focus} ---")
    graph_context_str = get_story_graph_context(character_focus)

    # 原来的 JSON 信息可以保留作为基础属性补充，也可以简化
    # 这里我们把图谱信息整合进去

    # 构建新的 Prompt 上下文部分
    character_section = f"""
    ### 👥 角色知识库与关系图谱 (GraphRAG)
    **基础档案**:
    {character_relevant_info} 

    **🕸️ 深度关系推理 (来自图数据库)**:
    {graph_context_str}
    """
    # 构建Prompt
    prompt_template = """
你是一位世界级的小说家，你的任务是基于所有给定的背景信息和创作指南，创作出故事的下一章。

{style_instruction_section}
---
这是上一章的内容，如果该章还是跟着上一章的内容，确保不要重复，要无缝衔接，如果上一章的故事情节暂且结束，可以不完全接着上一章的内容。
{previous_chapter_section}
---
{three_chapters_back_summary}
---
### 长期历史背景回顾 (AI记忆系统提供，前三个相关摘要)
{long_term_context}
---
{character_section}
---
### 本章创作核心指南
* **章节标题**: "{chapter_title}"
* **章节大纲**: {chapter_outline}
* **核心叙事目标**: {narrative_goals}
* **重点刻画角色**: {character_focus}
* **需要突出的主题**: {thematic_elements}
* **结构要求**: {structural_requirements}

### 输出要求
请严格按照指定的JSON格式输出，只包含章节草稿和需要调用的专家代理。

{format_instructions}
"""

    print("\n--- 🧠 Generating chapter draft with global style guide... ---")
    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "style_instruction_section", "previous_chapter_section",
                "three_chapters_back_summary", "long_term_context",
                "character_section", "chapter_title", "chapter_outline",
                "narrative_goals", "character_focus", "thematic_elements", "structural_requirements"
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({
            "style_instruction_section": style_instruction_section,
            "previous_chapter_section": previous_chapter_section,
            "three_chapters_back_summary": three_chapters_back_summary,
            "long_term_context": long_term_context,
            "character_section": character_section,
            "chapter_title": chapter_title,
            "chapter_outline": chapter_outline,
            "narrative_goals": ', '.join(narrative_goals),
            "character_focus": ', '.join(creative_brief.get('character_focus', [])),
            "thematic_elements": ', '.join(creative_brief.get('thematic_elements', [])),
            "structural_requirements": ', '.join(creative_brief.get('structural_requirements', []))
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_output = parse_json_with_filtering(filtered_content, parser)

        print(f"--- ✅ Successfully drafted Chapter: '{chapter_title}' ---")
        res_dict = parsed_output.model_dump()
        res_dict["required_agents"] = [
                "emotional_reader_agent", "rhythm_reader_agent", "immersion_reader_agent",
                "structural_novelist_agent", "foreshadowing_novelist_agent"
            ]
        return res_dict

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse output from LLM. Error: {e} ---")
        # 返回默认结构
        return {
            "initial_draft": "生成章节草稿时出现错误。",
            "required_agents": [
                "emotional_reader_agent", "rhythm_reader_agent", "immersion_reader_agent",
                "structural_novelist_agent", "foreshadowing_novelist_agent"
            ]
        }


# 在 rewriter_agent 函数中修复蒸馏部分

def rewriter_agent(state: StoryState) -> Dict[str, str]:
    """
    改写代理 v5.0
    - 使用输出解析器确保输出格式稳定
    """
    print("--- ✍️ 改写代理：根据所有建议并遵循总体风格进行修订... ---")

    draft_to_revise = state.get('initial_draft')
    all_suggestions = state.get('suggestions', [])
    style_guide = state.get("total_text_style", "无特定风格要求")

    if not draft_to_revise:
        return {"revised_draft": "错误：没有可修订的草稿。", "suggestions": []}
    if not all_suggestions:
        return {"revised_draft": draft_to_revise, "suggestions": []}

    # --- 反馈蒸馏 ---
    print("---  distilling feedback into a concise brief... ---")
    raw_suggestions_text = []
    for suggestion_group in all_suggestions:
        specialization = suggestion_group.get('specialization', '未知专家')
        suggestions_data = suggestion_group.get('suggestions', {})
        if isinstance(suggestions_data, dict):
            recommendations = suggestions_data.get('actionable_recommendations', [])
        else:
            recommendations = []
            print(f"⚠️ Warning: suggestions is not a dict: {suggestions_data}")
        if recommendations:
            raw_suggestions_text.append(f"来自 {specialization} 的建议:")
            for rec in recommendations:
                raw_suggestions_text.append(f"- {rec.get('suggestion', '')}")

    # 先将建议列表用换行符连接成一个单独的字符串变量
    suggestions_joined_text = "\n".join(raw_suggestions_text)

    # 创建蒸馏解析器
    distillation_parser = PydanticOutputParser(pydantic_object=DistillOutput)

    # 构建蒸馏提示模板
    distillation_prompt_template = """
你是一名顶级的总编辑。请阅读以下来自不同领域专家（情感、节奏、结构等）的反馈意见，
并将它们提炼、总结成一个给作家的、不超过5点的、清晰、可执行的修订指令列表。
请直接输出要点列表，不要有任何多余的客套话。

### 原始专家反馈:
{suggestions_joined_text}

### 输出要求
请严格按照指定的JSON格式输出。

{format_instructions}
"""

    llm = get_llm()

    try:
        # 构建蒸馏链
        distillation_prompt = PromptTemplate(
            template=distillation_prompt_template,
            input_variables=["suggestions_joined_text"],
            partial_variables={"format_instructions": distillation_parser.get_format_instructions()}
        )

        distillation_chain = distillation_prompt | llm
        raw_response0 = distillation_chain.invoke({
            "suggestions_joined_text": suggestions_joined_text
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response0.content)
        parsed_distill = parse_json_with_filtering(filtered_content, distillation_parser)
        distilled_brief = parsed_distill.distilled_brief

        print("--- ✅ Feedback distilled successfully. ---")
        print(f"--- Distilled Brief ---\n{distilled_brief}\n-------------------")

    except Exception as e:
        print(f"--- ❌ Error: Failed to distill feedback. Error: {e} ---")
        # 如果蒸馏失败，使用原始建议
        distilled_brief = ["基于专家建议进行修订"]
        print("--- ⚠️ Using fallback distilled brief ---")

    # 创建重写解析器
    parser = PydanticOutputParser(pydantic_object=RewriterOutput)

    # 使用蒸馏后的简洁指令构建最终的重写提示
    rewrite_prompt_template = """
你是一家著名出版社的资深编辑。你的核心任务是根据一份"总修订指令"来重写原始草稿。

### 故事总体风格指南 (最终稿件必须符合此风格)
**{style_guide}**

#### 如何在修订中应用风格的示例:
* **指定的风格要求**: "语言简洁有力，聚焦于角色的感受和行动，而不是景物本身。"
* **原始文本 (不佳)**: "他走在森林里，高大的树木遮蔽了天空，阳光从树叶的缝隙中斑驳地洒下，地上的落叶发出沙沙的声响。"
* **修订后文本 (优秀)**: "森林的巨木让他感到压抑和渺小。他踩在枯叶上发出的脆响，是这片死寂中唯一的声音。偶尔有阳光穿透林冠，带来一丝转瞬即逝的暖意，却驱不散他骨子里的寒冷。"

---

### 原始草稿:
{draft_to_revise}

---

### 总修订指令 (需要整合的要点):
{distilled_brief_text}

---

### 输出要求
请严格按照指定的JSON格式输出修订后的草稿。

{format_instructions}
"""

    print("--- 🧠 Generating revised draft based on distilled brief and global style... ---")

    try:
        # 将蒸馏后的简要转换为字符串
        distilled_brief_text = "\n".join([f"- {point}" for point in distilled_brief])

        prompt = PromptTemplate(
            template=rewrite_prompt_template,
            input_variables=["style_guide", "draft_to_revise", "distilled_brief_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({
            "style_guide": style_guide,
            "draft_to_revise": draft_to_revise,
            "distilled_brief_text": distilled_brief_text
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_output = parse_json_with_filtering(filtered_content, parser)

        print("--- ✅ 改写代理：草稿已成功修订。 ---")
        res_dict = parsed_output.model_dump()
        res_dict["suggestions"] = []
        return res_dict

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse output from LLM. Error: {e} ---")
        # 返回默认结构
        return {"revised_draft": draft_to_revise, "suggestions": []}


# 在 writer.py 文件末尾添加以下测试函数

def writer_agent1():
    """测试作家 Agent"""
    print("🧪 测试作家 Agent...")

    # 创建模拟状态
    test_state: StoryState = {
        "current_chapter_index": 1,
        "chapter_title": "神秘的信件",
        "chapter_outline": "主角收到神秘信件，开始冒险之旅",
        "creative_brief": {
            "narrative_goals": ["建立主角性格", "引入主要冲突"],
            "character_focus": ["小明"],
            "thematic_elements": ["勇气", "成长"],
            "structural_requirements": ["三幕式结构"]
        },
        "full_text_history": [],
        "summary_history": [],
        "total_text_style": "以情节和主角内心活动为绝对核心，快节奏推进",
        "agent_flags": {}
    }

    result = writer_agent(test_state)
    print("作家 Agent 测试结果:")
    print(f"草稿长度: {len(result.get('initial_draft', ''))} 字符")
    print(f"所需代理: {result.get('required_agents', [])}")
    print(f"包含初始草稿: {'initial_draft' in result}")
    print(f"包含所需代理: {'required_agents' in result}")
    return True


def rewriter_agent1():
    """测试改写 Agent"""
    print("🧪 测试改写 Agent...")

    # 创建模拟状态
    test_state: StoryState = {
        "initial_draft": "小明是一个普通的学生。某天，他收到了一封神秘的信件。信件没有署名，只有一行字：'你的命运在等待着你。'小明感到困惑，但内心却有一丝莫名的兴奋。",
        "suggestions": [
            {
                "specialization": "情感共鸣专家",
                "suggestions": {
                    "actionable_recommendations": [
                        {
                            "priority": "中",
                            "location": "第一段",
                            "suggestion": "增加小明收到信件时的心理活动描写",
                            "expected_impact": "增强情感共鸣"
                        }
                    ]
                }
            },
            {
                "specialization": "节奏与悬念专家",
                "suggestions": {
                    "actionable_recommendations": [
                        {
                            "priority": "高",
                            "location": "信件内容",
                            "suggestion": "让信件内容更加神秘，增加悬念",
                            "expected_impact": "提升读者兴趣"
                        }
                    ]
                }
            }
        ],
        "total_text_style": "语言简洁有力，聚焦于角色的感受和行动",
        "agent_flags": {}
    }

    result = rewriter_agent(test_state)
    print("改写 Agent 测试结果:")
    print(f"修订草稿长度: {len(result.get('revised_draft', ''))} 字符")
    print(f"建议列表: {result.get('suggestions', [])}")
    print(f"包含修订草稿: {'revised_draft' in result}")
    print(f"包含建议字段: {'suggestions' in result}")
    return True


def writer_error_handling():
    """测试作家 Agent 错误处理"""
    print("🧪 测试作家 Agent 错误处理...")

    # 创建模拟状态 - 缺少必要字段
    test_state: StoryState = {
        "current_chapter_index": 1,
        "chapter_title": "测试章节",
        # 缺少 chapter_outline 和 creative_brief
        "agent_flags": {}
    }

    result = writer_agent(test_state)
    print("作家 Agent 错误处理测试结果:")
    print(f"结果类型: {type(result)}")
    print(f"包含默认结构: {'initial_draft' in result and 'required_agents' in result}")
    return True


def rewriter_empty_suggestions():
    """测试改写 Agent 无建议情况"""
    print("🧪 测试改写 Agent 无建议情况...")

    # 创建模拟状态 - 没有建议
    test_state: StoryState = {
        "initial_draft": "这是一个测试草稿。",
        "suggestions": [],  # 空建议列表
        "total_text_style": "测试风格",
        "agent_flags": {}
    }

    result = rewriter_agent(test_state)
    print("改写 Agent 无建议测试结果:")
    print(f"修订草稿与原始相同: {result.get('revised_draft') == '这是一个测试草稿。'}")
    print(f"建议列表为空: {result.get('suggestions') == []}")
    return True


def ewriter_no_draft():
    """测试改写 Agent 无草稿情况"""
    print("🧪 测试改写 Agent 无草稿情况...")

    # 创建模拟状态 - 没有草稿
    test_state: StoryState = {
        "initial_draft": None,  # 无草稿
        "suggestions": [{"specialization": "测试专家", "suggestions": {}}],
        "total_text_style": "测试风格",
        "agent_flags": {}
    }

    result = rewriter_agent(test_state)
    print("改写 Agent 无草稿测试结果:")
    print(f"返回错误信息: {'错误' in result.get('revised_draft', '')}")
    print(f"建议列表为空: {result.get('suggestions') == []}")
    return True


def run_all_writer_tests():
    """运行所有 Writer Agents 测试"""
    print("🚀 开始测试 Writer Agents...\n")
    writer_agent1()
    rewriter_agent1()
    writer_error_handling()
    rewriter_empty_suggestions()
    ewriter_no_draft()


# 在文件末尾添加以下代码来运行测试
if __name__ == "__main__":
    # 如果直接运行这个文件，执行所有测试
    run_all_writer_tests()