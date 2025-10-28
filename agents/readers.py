
import json
import os
import dotenv
from pydantic import BaseModel, Field
import re

from helper import filter_think_tags, parse_json_with_filtering
from state import StoryState, Suggestion
from utils import get_llm

dotenv.load_dotenv()
# 确保 OPENAI_API_KEY 已设置

from typing import TypedDict, List, Any, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from langchain.prompts import PromptTemplate


# ==================== 辅助函数 ====================

def create_default_feedback(agent_id: str, specialization: str, error_msg: str = "") -> Dict[str, Any]:
    """创建默认的反馈结构"""
    default_feedback = {
        "agent_id": agent_id,
        "specialization": specialization,
        "suggestions": {
            "qualitative_feedback": {
                "strengths": ["分析过程中出现异常"],
                "weaknesses": [error_msg if error_msg else "无法完成分析"]
            },
            "actionable_recommendations": [
                {
                    "priority": "高",
                    "location": "整体",
                    "suggestion": "请检查输入内容格式或重新运行分析",
                    "expected_impact": "确保分析流程正常进行"
                }
            ]
        }
    }
    return default_feedback


# ==================== 定义 Pydantic 模型 ====================

class QualitativeFeedback(BaseModel):
    strengths: List[str] = Field(description="识别出的优点")
    weaknesses: List[str] = Field(description="发现的改进点")


class ActionableRecommendation(BaseModel):
    priority: str = Field(description="优先级：高/中/低")
    location: str = Field(description="建议修改的具体位置，如：第X段")
    suggestion: str = Field(description="具体的、可操作的修改建议")  # 确保这个字段名与 JSON 中的一致
    expected_impact: str = Field(description="该修改预计会带来的提升效果")  # 确保这个字段名与 JSON 中的一致


class SuggestionsModel(BaseModel):
    qualitative_feedback: QualitativeFeedback
    actionable_recommendations: List[ActionableRecommendation]


class EmotionalAnalysis(BaseModel):
    agent_id: str = Field(default="Reader_1_Emotional", description="代理ID")
    specialization: str = Field(default="情感共鸣专家", description="专业领域")
    suggestions: SuggestionsModel = Field(description="分析建议")


class RhythmAnalysis(BaseModel):
    agent_id: str = Field(default="Reader_2_Rhythm", description="代理ID")
    specialization: str = Field(default="节奏与悬念专家", description="专业领域")
    suggestions: SuggestionsModel = Field(description="分析建议")


class ImmersionAnalysis(BaseModel):
    agent_id: str = Field(default="Reader_3_Immersion", description="代理ID")
    specialization: str = Field(default="世界观沉浸专家", description="专业领域")
    suggestions: SuggestionsModel = Field(description="分析建议")


# ==================== 情感共鸣专家 Agent ====================

def emotional_reader_agent(state: StoryState) -> Dict[str, Any]:
    """
    情感共鸣专家 Agent (Emotional Reader Agent) - v3.0
    使用 LangChain JSONOutputParser 确保输出格式稳定
    """
    print("--- 🧐 Emotional Reader Agent: Analyzing draft... ---")
    if state["agent_flags"].get("emotional_reader_agent") == 0:
        return {}

    chapter_draft = state.get('initial_draft')
    revision_brief = state.get('revision_brief')

    # --- 构建针对性修订指南 ---
    targeted_guidance = ""
    if revision_brief:
        feedback = revision_brief.get("emotional", {})
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
        error_feedback = create_default_feedback(
            "Reader_1_Emotional",
            "情感共鸣专家",
            "未在状态中找到可供分析的章节草稿。"
        )
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=EmotionalAnalysis)

    prompt_template = """
你是一名AI叙事分析师，专业领域是"情感共鸣专家"。你的任务是基于提供的章节草稿，进行深入的情感分析,并给出修改建议。

{targeted_guidance}

**章节草稿**:
{chapter_draft}

**核心分析维度**:
1.  **情感真实性 (Emotional Authenticity)**: 角色的情感反应是否可信、自然？
2.  **角色共情度 (Character Empathy)**: 读者是否容易与角色的情感产生连接和共鸣？
3.  **心理描写深度 (Psychological Depth)**: 角色的内心世界和心理活动是否描绘得足够深刻？
4.  **情感变化弧线 (Emotional Arc)**: 本章中角色的情感变化是否清晰、完整且有说服力？

**输出要求**:
1. 禁止添加任何解释、思考过程、备注或说明文字（包括但不限于"好的，<think></think>, 我现在需要..."等类似文本）。
2. 必须严格按照{format_instructions}中指定的JSON格式输出
3. 只返回JSON内容，不添加任何额外解释、说明或思考过程
4. 确保包含所有必填字段，特别是"suggestions"字段，其包含"qualitative_feedback"和"actionable_recommendations"子字段
5. "qualitative_feedback"需要包含"strengths"和"weaknesses"列表
6. "actionable_recommendations"需要是一个包含多个建议的列表，每个建议包含"priority"、"location"、"suggestion"和"expected_impact"
7. 不要输出<think></think>直接的内容

{format_instructions}
"""

    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["targeted_guidance", "chapter_draft"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 获取原始响应
        chain = prompt | llm
        raw_response = chain.invoke({
            "targeted_guidance": targeted_guidance,
            "chapter_draft": chapter_draft
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_feedback = parse_json_with_filtering(filtered_content, parser)

        print("--- ✅ Emotional Reader Agent: Analysis complete. ---")
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(parsed_feedback.model_dump())
        return {"suggestions": current_suggestions}

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse feedback from LLM. Error: {e} ---")
        error_feedback = create_default_feedback(
            "Reader_1_Emotional",
            "情感共鸣专家",
            f"分析过程中出现错误: {str(e)}"
        )
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}


# ==================== 节奏与悬念专家 Agent ====================

def rhythm_reader_agent(state: StoryState) -> Dict[str, Any]:
    """
    节奏与悬念专家 Agent (Rhythm Reader Agent) - v3.0
    使用 LangChain JSONOutputParser 确保输出格式稳定
    """
    if state["agent_flags"].get("rhythm_reader_agent") == 0:
        return {}
    print("--- 🎭 Rhythm Reader Agent: Analyzing pacing and suspense... ---")

    chapter_draft = state.get('initial_draft')
    revision_brief = state.get('revision_brief')

    # --- 构建针对性修订指南 ---
    targeted_guidance = ""
    if revision_brief:
        feedback = revision_brief.get("rhythm", {})
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
        error_feedback = create_default_feedback(
            "Reader_2_Rhythm",
            "节奏与悬念专家",
            "未在状态中找到可供分析的章节草稿。"
        )
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=RhythmAnalysis)

    prompt_template = """
你是一名AI叙事分析师，专业领域是"节奏与悬念专家"。你的任务是基于提供的章节草稿，进行深入的节奏和悬念分析。

{targeted_guidance}

**章节草稿**:
{chapter_draft}

**核心分析维度**:
1.  **节奏控制力 (Pacing Control)**: 章节的节奏变化是否合理？快慢交替是否自然？
2.  **悬念密度 (Suspense Density)**: 悬念设置是否足够且分布合理？
3.  **章节结构合理性 (Chapter Structure)**: 开头、发展、高潮、结尾的结构是否清晰？
4.  **高潮效果强度 (Climax Effectiveness)**: 高潮部分是否具有足够的情感冲击力和戏剧性？

**输出要求**:
1. 禁止添加任何解释、思考过程、备注或说明文字（包括但不限于"好的，<think></think>, 我现在需要..."等类似文本）。
2. 必须严格按照{format_instructions}中指定的JSON格式输出
3. 只返回JSON内容，不添加任何额外解释、说明或思考过程
4. 确保包含所有必填字段，特别是"suggestions"字段，其包含"qualitative_feedback"和"actionable_recommendations"子字段
5. "qualitative_feedback"需要包含"strengths"和"weaknesses"列表
6. "actionable_recommendations"需要是一个包含多个建议的列表，每个建议包含"priority"、"location"、"suggestion"和"expected_impact"

{format_instructions}
"""

    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["targeted_guidance", "chapter_draft"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 获取原始响应
        chain = prompt | llm
        raw_response = chain.invoke({
            "targeted_guidance": targeted_guidance,
            "chapter_draft": chapter_draft
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        #print(f"--- 🔍 Rhythm Reader Raw Response: {filtered_content} ---")  # 调试输出

        # 修复可能的字段名不匹配问题
        if "actionable_recommendations" not in filtered_content and "actionable_recommendation" in filtered_content:
            filtered_content = filtered_content.replace('"actionable_recommendation"', '"actionable_recommendations"')

        parsed_feedback = parse_json_with_filtering(filtered_content, parser)

        print("--- ✅ Rhythm Reader Agent: Analysis complete. ---")
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(parsed_feedback.model_dump())
        return {"suggestions": current_suggestions}

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse feedback from LLM. Error: {e} ---")
        # 打印原始响应以便调试
        if 'raw_response' in locals():
            print(f"--- 🔍 Raw response for debugging: {raw_response.content} ---")

        error_feedback = create_default_feedback(
            "Reader_2_Rhythm",
            "节奏与悬念专家",
            f"分析过程中出现错误: {str(e)}"
        )
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}


# ==================== 世界观沉浸专家 Agent ====================

def immersion_reader_agent(state: StoryState) -> Dict[str, Any]:
    """
    世界观沉浸专家 Agent (Immersion Reader Agent) - v3.0
    使用 LangChain JSONOutputParser 确保输出格式稳定
    """
    if state["agent_flags"].get("immersion_reader_agent") == 0:
        return {}
    print("--- 🌍 Immersion Reader Agent: Analyzing world-building... ---")

    chapter_draft = state.get('initial_draft')
    revision_brief = state.get('revision_brief')

    # --- 构建针对性修订指南 ---
    targeted_guidance = ""
    if revision_brief:
        feedback = revision_brief.get("immersion", {})
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
        error_feedback = create_default_feedback(
            "Reader_3_Immersion",
            "世界观沉浸专家",
            "未在状态中找到可供分析的章节草稿。"
        )
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=ImmersionAnalysis)

    prompt_template = """
你是一名AI叙事分析师，专业领域是"世界观沉浸专家"。你的任务是进行深入的世界观一致性和沉浸感分析。

{targeted_guidance}

**章节草稿**:
{chapter_draft}

**核心分析维度**:
1.  **世界观一致性 (World Consistency)**: 环境描写、设定元素是否与已建立的世界观保持一致？
2.  **细节丰富度 (Detail Richness)**: 感官细节（视觉、听觉、嗅觉、触觉等）是否足够丰富？
3.  **沉浸感强度 (Immersion Level)**: 读者是否能完全沉浸在故事世界中？
4.  **逻辑连贯性 (Logical Coherence)**: 情节发展和角色行为是否符合世界观的内在逻辑？

**输出要求**:
1. 禁止添加任何解释、思考过程、备注或说明文字（包括但不限于"好的，<think></think>, 我现在需要..."等类似文本）。
2. 必须严格按照{format_instructions}中指定的JSON格式输出
3. 只返回JSON内容，不添加任何额外解释、说明或思考过程
4. 确保包含所有必填字段，特别是"suggestions"字段，其包含"qualitative_feedback"和"actionable_recommendations"子字段
5. "qualitative_feedback"需要包含"strengths"和"weaknesses"列表
6. "actionable_recommendations"需要是一个包含多个建议的列表，每个建议包含"priority"、"location"、"suggestion"和"expected_impact"

{format_instructions}
"""

    llm = get_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["targeted_guidance", "chapter_draft"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 获取原始响应
        chain = prompt | llm
        raw_response = chain.invoke({
            "targeted_guidance": targeted_guidance,
            "chapter_draft": chapter_draft
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_feedback = parse_json_with_filtering(filtered_content, parser)

        print("--- ✅ Immersion Reader Agent: Analysis complete. ---")
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(parsed_feedback.model_dump())
        return {"suggestions": current_suggestions}

    except Exception as e:
        print(f"--- ❌ Error: Failed to parse feedback from LLM. Error: {e} ---")
        error_feedback = create_default_feedback(
            "Reader_3_Immersion",
            "世界观沉浸专家",
            f"分析过程中出现错误: {str(e)}"
        )
        current_suggestions = state.get("suggestions", [])
        current_suggestions.append(error_feedback)
        return {"suggestions": current_suggestions}


def reader_agent1():
    """测试情感共鸣专家 Agent 的输出解析"""
    print("🧪 测试情感共鸣专家 Agent...")

    # 创建模拟状态
    test_state: StoryState = {
        "initial_draft": "这是一个测试章节内容。角色感到非常悲伤，然后逐渐变得坚强。",
        "agent_flags": {"emotional_reader_agent": 1},
        "suggestions": []
    }
    result = emotional_reader_agent(test_state)
    print(result)


def reader_agent2():
    """测试世界观沉浸专家 Agent 的输出解析"""
    print("🧪 测试世界观沉浸专家 Agent...")

    # 创建模拟状态
    test_state: StoryState = {
        "initial_draft": "这是一个测试章节内容。角色感到非常悲伤，然后逐渐变得坚强。",
        "agent_flags": {"immersion_reader_agent": 1},
        "suggestions": []
    }
    result = immersion_reader_agent(test_state)
    print(result)


def reader_agent3():
    """测试节奏与悬念专家 Agent 的输出解析"""
    print("🧪 测试节奏与悬念专家 Agent...")

    # 创建模拟状态
    test_state: StoryState = {
        "initial_draft": "这是一个测试章节内容。角色感到非常悲伤，然后逐渐变得坚强。",
        "agent_flags": {"rhythm_reader_agent": 1},
        "suggestions": []
    }
    result = rhythm_reader_agent(test_state)
    print(result)


if __name__ == "__main__":
    reader_agent1()
    reader_agent2()
    reader_agent3()
