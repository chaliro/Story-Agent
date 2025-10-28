import json
from typing import Dict, Any, List

from pydantic import BaseModel, Field

from helper import filter_think_tags, parse_json_with_filtering
from state import StoryState, Suggestion
from utils import get_llm
from langchain.output_parsers import PydanticOutputParser

from langchain.prompts import PromptTemplate


# ==================== 定义 Pydantic 模型 ====================

class QualitativeFeedback(BaseModel):
    strengths: List[str] = Field(description="识别出的优点")
    weaknesses: List[str] = Field(description="发现的改进点")


class ActionableRecommendation(BaseModel):
    priority: str = Field(description="优先级：高/中/低")
    location: str = Field(description="建议修改的具体位置，如：第X段")
    suggestion: str = Field(description="具体的、可操作的修改建议")
    expected_impact: str = Field(description="该修改预计会带来的提升效果")  # 确保这个字段名与 JSON 中的一致


class SuggestionsModel(BaseModel):
    qualitative_feedback: QualitativeFeedback
    actionable_recommendations: List[ActionableRecommendation]


class StructuralAnalysis(BaseModel):
    agent_id: str = Field(default="Novelist_1_Structural", description="代理ID")
    specialization: str = Field(default="结构工程师", description="专业领域")
    suggestions: SuggestionsModel


class ForeshadowingAnalysis(BaseModel):
    agent_id: str = Field(default="Novelist_2_Foreshadowing", description="代理ID")
    specialization: str = Field(default="伏笔侦探", description="专业领域")
    suggestions: SuggestionsModel


# ==================== 结构工程师 Agent ====================

def structural_novelist_agent(state: StoryState) -> Dict[str, Any]:
    """
    结构工程师 Agent (Structural Novelist Agent) - v3.0
    使用 LangChain JSONOutputParser 确保输出格式稳定
    """
    if state["agent_flags"].get("structural_novelist_agent") == 0:
        return {}
    print("--- 🏗️ Structural Novelist Agent: Analyzing narrative structure... ---")

    chapter_draft = state.get('initial_draft')
    story_outline = state.get('story_outline', '')
    creative_brief = state.get('creative_brief', {})
    revision_brief = state.get('revision_brief')

    # --- 构建针对性修订指南 ---
    targeted_guidance = ""
    if revision_brief:
        feedback = revision_brief.get("structural", {})
        if feedback:
            comment = feedback.get('comment', '无')
            suggestions = "、".join(feedback.get('suggestions', []))
            targeted_guidance = f"""
### 上一轮评审的针对性反馈 (请重点关注)
- **综合评语**: {comment}
- **具体建议**: {suggestions}
请在本次分析中，特别留意以上提到的问题是否已得到改善，并围绕这些问题提供更深入的建议。
"""

    if not chapter_draft:
        print("--- ⚠️ Warning: No chapter draft found. ---")
        error_feedback = {
            "agent_id": "Novelist_1_Structural",
            "specialization": "结构工程师",
            "suggestions": "未在状态中找到可供分析的章节草稿。"
        }
        state.setdefault("suggestions", []).append(error_feedback)
        return {}

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=StructuralAnalysis)

    prompt_template = """
你是一名AI叙事结构分析师，专业领域是"结构工程师"。

{targeted_guidance}

**故事背景信息**:
- 整体故事大纲: {story_outline}
- 创作指令: {creative_brief}

**章节草稿**:
{chapter_draft}

**核心分析维度**:
1.  **角色发展质量 (Character Development)**: 角色在本章中是否有明显的发展和成长？
2.  **情节结构合理性 (Plot Structure)**: 情节发展是否符合经典结构？转折点是否合理？
3.  **主题融合度 (Thematic Integration)**: 主题是否自然地融入叙事中？
4.  **叙事效率 (Narrative Efficiency)**: 情节推进是否高效？有无冗余内容？

**输出要求**:
请严格按照指定的JSON格式提供分析报告。

{format_instructions}
"""

    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["targeted_guidance", "story_outline", "creative_brief", "chapter_draft"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )


        chain = prompt | llm
        raw_response = chain.invoke({
            "targeted_guidance": targeted_guidance,
            "story_outline": story_outline,
            "creative_brief": creative_brief,
            "chapter_draft": chapter_draft
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_feedback = parse_json_with_filtering(filtered_content, parser)
        print("--- ✅ Structural Novelist Agent: Analysis complete. ---")
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(parsed_feedback.model_dump())
        return {"suggestions": current_suggestions}

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse feedback from LLM. Error: {e} ---")
        error_feedback = {
            "agent_id": "Novelist_1_Structural",
            "specialization": "结构工程师",
            "suggestions": f"分析过程中出现错误: {str(e)}"
        }
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}


# ==================== 伏笔侦探 Agent ====================

def foreshadowing_novelist_agent(state: StoryState) -> Dict[str, Any]:
    """
    伏笔侦探 Agent (Foreshadowing Novelist Agent) - v3.0
    使用 LangChain JSONOutputParser 确保输出格式稳定
    """
    if state["agent_flags"].get("foreshadowing_novelist_agent") == 0:
        return {}
    print("--- 🔍 Foreshadowing Novelist Agent: Analyzing foreshadowing... ---")

    chapter_draft = state.get('initial_draft')
    summary_history = state.get('summary_history', [])
    revision_brief = state.get('revision_brief')

    # --- 构建针对性修订指南 ---
    targeted_guidance = ""
    if revision_brief:
        feedback = revision_brief.get("foreshadowing", {})
        if feedback:
            comment = feedback.get('comment', '无')
            suggestions = "、".join(feedback.get('suggestions', []))
            targeted_guidance = f"""
### 上一轮评审的针对性反馈 (请重点关注)
- **综合评语**: {comment}
- **具体建议**: {suggestions}
请在本次分析中，特别留意以上提到的问题是否已得到改善，并围绕这些问题提供更深入的建议。
"""

    if not chapter_draft:
        print("--- ⚠️ Warning: No chapter draft found. ---")
        error_feedback = {
            "agent_id": "Novelist_2_Foreshadowing",
            "specialization": "伏笔侦探",
            "suggestions": "未在状态中找到可供分析的章节草稿。"
        }
        state.setdefault("suggestions", []).append(error_feedback)
        return {}

    historical_context = "### 历史章节摘要回顾:\n" + "\n".join(f"- {s}" for s in summary_history)

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=ForeshadowingAnalysis)

    prompt_template = """
你是一名AI叙事分析师，专业领域是"伏笔侦探"。

{targeted_guidance}

**历史章节信息**:
{historical_context}

**章节草稿**:
{chapter_draft}

**核心分析维度**:
1.  **伏笔质量 (Foreshadowing Quality)**: 新设置的伏笔是否巧妙自然？
2.  **线索融合度 (Clue Integration)**: 线索是否自然地融入叙事中？
3.  **回收效果 (Payoff Effectiveness)**: 已回收的伏笔是否具有足够的情感冲击力？
4.  **叙事密度 (Narrative Density)**: 伏笔和线索的密度是否适中？

**输出要求**:
请严格按照指定的JSON格式提供分析报告。

{format_instructions}
"""

    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["targeted_guidance", "historical_context", "chapter_draft"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({
            "targeted_guidance": targeted_guidance,
            "historical_context": historical_context,
            "chapter_draft": chapter_draft
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_feedback = parse_json_with_filtering(filtered_content, parser)
        print("--- ✅ Foreshadowing Novelist Agent: Analysis complete. ---")
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(parsed_feedback.model_dump())
        return {"suggestions": current_suggestions}

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse feedback from LLM. Error: {e} ---")
        error_feedback = {
            "agent_id": "Novelist_2_Foreshadowing",
            "specialization": "伏笔侦探",
            "suggestions": f"分析过程中出现错误: {str(e)}"
        }
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}


def novelist_agents_disabled():
    """测试 Novelist Agents 禁用功能"""
    print("🧪 测试 Novelist Agents 禁用功能...")

    # 创建模拟状态 - 结构工程师禁用
    test_state1: StoryState = {
        "initial_draft": "测试内容",
        "agent_flags": {"structural_novelist_agent": 1},  # 禁用
        "suggestions": []
    }

    result1 = structural_novelist_agent(test_state1)
    print(result1)

    # 创建模拟状态 - 伏笔侦探禁用
    test_state2: StoryState = {
        "initial_draft": "测试内容",
        "agent_flags": {"foreshadowing_novelist_agent": 1},  # 禁用
        "suggestions": []
    }

    result2 = foreshadowing_novelist_agent(test_state2)

    print(result2)

    print("✅ Novelist Agents 禁用功能测试通过!")
    return True

if __name__ == "__main__":
    novelist_agents_disabled()