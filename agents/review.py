import json
from typing import Dict, Any, List

from pydantic import BaseModel, Field

from helper import filter_think_tags, parse_json_with_filtering
from state import CommitteeDecision, StoryState
from utils import get_evaluation_llm
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


# ==================== 定义 Review Committee 的 Pydantic 模型 ====================

class DimensionScore(BaseModel):
    score: int = Field(description="维度评分 1-10分")
    comment: str = Field(description="综合评语")
    suggestions: List[str] = Field(description="基于专家评估的改进建议")


class CreativeBrief(BaseModel):
    narrative_goals: List[str] = Field(description="基于评估的叙事目标")
    character_focus: List[str] = Field(description="基于评估的角色焦点")
    thematic_elements: List[str] = Field(description="基于评估的主题元素")
    structural_requirements: List[str] = Field(description="基于评估的结构要求")


class NextChapterDirection(BaseModel):
    chapter_index: int = Field(description="下一章索引")
    chapter_title: str = Field(description="基于故事发展的下一章标题")
    chapter_outline: str = Field(description="基于当前章节评估的下一章大纲")
    creative_brief: CreativeBrief = Field(description="下一章创作指令")


class ReviewCommitteeDecision(BaseModel):
    overall_verdict: str = Field(description="总体裁决: PUBLISH|FORCE_PUBLISH|REVISE")
    dimension_scores: Dict[str, DimensionScore] = Field(description="五个维度的评分和评语")
    next_chapter_creative_direction: NextChapterDirection = Field(description="下一章创作方向")
    dimensions_requiring_revision: List[str] = Field(description="需要修改的维度列表")


# ==================== 审稿人委员会 Agent ====================

def review_agent(state: StoryState) -> Dict[str, Any]:
    """
    审稿人委员会 Agent (Review Committee Agent) - 使用输出解析器版本

    基于专家评估结果、修订稿和故事上下文，生成最终决策和下一章创作方向，
    并确定下一轮需要哪些agent参与修改。

    Args:
        state (StoryState): 当前的 StoryState，包含所有必要信息

    Returns:
        Dict[str, Any]: 包含 committee_decision 和 required_agents 的字典
    """
    print("--- 🏛️ Review Committee: Evaluating revised draft and making final decision... ---")

    # 从状态中提取必要信息
    title = state.get('title', '')
    story_outline = state.get('story_outline', '')
    current_chapter_index = state.get('current_chapter_index', 0)
    chapter_title = state.get('chapter_title', '')
    chapter_outline = state.get('chapter_outline', '')
    creative_brief = state.get('creative_brief', {})
    revised_draft = state.get('revised_draft', '')
    expert_evaluations = state.get('expert_evaluations', [])
    full_text_history = state.get('full_text_history', [])
    summary_history = state.get('summary_history', [])

    # 创建默认的错误决策结构
    error_decision: CommitteeDecision = {
        "overall_verdict": "REVISE",
        "dimension_scores": {},
        "next_chapter_creative_direction": {
            "chapter_index": current_chapter_index + 1,
            "chapter_title": f"第{current_chapter_index + 1}章",
            "chapter_outline": "待确定",
            "creative_brief": {}
        }
    }

    if not revised_draft:
        print("--- ⚠️ Warning: No revised draft found for review. ---")
        return {
            "committee_decision": error_decision,
            "required_agents": ["emotional_reader_agent", "rhythm_reader_agent", "immersion_reader_agent",
                                "structural_novelist_agent", "foreshadowing_novelist_agent"]
        }

    if not expert_evaluations:
        print("--- ⚠️ Warning: No expert evaluations found. ---")
        return {
            "committee_decision": error_decision,
            "required_agents": ["emotional_reader_agent", "rhythm_reader_agent", "immersion_reader_agent",
                                "structural_novelist_agent", "foreshadowing_novelist_agent"]
        }

    # 构建故事上下文信息
    story_context = f"""
小说标题: {title}
故事大纲: {story_outline}
已发布章节数: {len(full_text_history)}
"""

    # 构建专家评估汇总
    expert_summary = "专家评估汇总:\n"
    for eval in expert_evaluations:
        expert_summary += f"\n## {eval['specialization']} ({eval['agent_id']})\n"
        for dimension, score in eval['dimension_scores'].items():
            comment = eval['evaluation_comments'].get(dimension, '')
            expert_summary += f"- {dimension}: {score}/10 - {comment}\n"

    # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=ReviewCommitteeDecision)

    # 构建高度结构化的Prompt
    prompt_template = """
你是一名AI审稿委员会主席，负责基于专家评估结果和故事上下文，对修订稿做出最终决策，并为下一章提供创作方向。

**故事上下文**:
{story_context}

**当前章节信息**:
- 章节编号: 第{current_chapter_index}章
- 章节标题: {chapter_title}
- 章节大纲: {chapter_outline}
- 创作指令: {creative_brief}

**专家评估结果**:
{expert_summary}

**待审阅修订稿内容**:
{revised_draft}

**你的任务**:
基于以上所有信息，特别是专家评估结果，做出综合决策并为下一章提供创作方向。你需要：

1. **总体裁决 (Overall Verdict)**:
   - "PUBLISH": 质量优秀，直接发布
   - "FORCE_PUBLISH": 质量尚可，强制发布（用于控制重写次数）
   - "REVISE": 需要重大修改

2. **维度评分 (Dimension Scores)**:
   基于专家评估，为以下五个维度提供综合评分、评语和改进建议：
   - emotional (情感表达): 综合情感可及性和情感共鸣方面的评估
   - rhythm (节奏把控): 综合语言节奏和阅读节奏方面的评估
   - foreshadowing (伏笔设置): 综合伏笔艺术表现力方面的评估
   - immersion (沉浸感): 综合感官沉浸、世界观融合和代入感方面的评估
   - structural (结构安排): 综合叙事结构、连贯性和创新性方面的评估

3. **下一章创作方向 (Next Chapter Creative Direction)**:
   基于当前章节的评估结果和故事发展，为下一章提供详细的创作指导

4. **需要修改的维度 (Dimensions Requiring Revision)**:
   根据评分和评语，确定哪些维度需要进一步修改。请列出需要修改的维度名称（从以下五个中选择：emotional, rhythm, foreshadowing, immersion, structural）
   请注意如果评分是8分及以上就无须修改，如果是6分和7分你需要思考是否有要修改的必要，如果是6分以下，一定需要重写。

**输出要求**:
请严格按照指定的JSON格式提供决策报告。

{format_instructions}
"""

    llm = get_evaluation_llm()

    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["story_context", "current_chapter_index", "chapter_title",
                             "chapter_outline", "creative_brief", "expert_summary", "revised_draft"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm
        raw_response = chain.invoke({
            "story_context": story_context,
            "current_chapter_index": current_chapter_index,
            "chapter_title": chapter_title,
            "chapter_outline": chapter_outline,
            "creative_brief": creative_brief,
            "expert_summary": expert_summary,
            "revised_draft": revised_draft
        })

        # 过滤思考标签并解析
        filtered_content = filter_think_tags(raw_response.content)
        parsed_decision = parse_json_with_filtering(filtered_content, parser)

        print("--- ✅ Review Committee: Decision made successfully. ---")
        print(f"Overall Verdict: {parsed_decision.overall_verdict}")

        # 打印各维度评分
        for dimension, score_info in parsed_decision.dimension_scores.items():
            print(f"{dimension}: {score_info.score}分")

        # 根据需要修改的维度确定需要哪些agent
        dimensions_requiring_revision = parsed_decision.dimensions_requiring_revision
        required_agents = []

        # 映射维度到对应的agent
        dimension_to_agent = {
            "emotional": "emotional_reader_agent",
            "rhythm": "rhythm_reader_agent",
            "immersion": "immersion_reader_agent",
            "structural": "structural_novelist_agent",
            "foreshadowing": "foreshadowing_novelist_agent"
        }

        for dimension in dimensions_requiring_revision:
            if dimension in dimension_to_agent:
                required_agents.append(dimension_to_agent[dimension])

        # 如果总体裁决是REVISE但没有指定需要修改的维度，默认使用所有agent
        if parsed_decision.overall_verdict == 'REVISE' and not required_agents:
            required_agents = list(dimension_to_agent.values())

        print(f"需要修改的维度: {dimensions_requiring_revision}")
        print(f"需要执行的Agent: {required_agents}")

        # 将 Pydantic 模型转换为字典格式
        decision_dict = parsed_decision.model_dump()

        # 提取CommitteeDecision部分（与原始结构保持一致）
        committee_decision: CommitteeDecision = {
            "overall_verdict": decision_dict["overall_verdict"],
            "dimension_scores": {
                dim: {
                    "score": info["score"],
                    "comment": info["comment"],
                    "suggestions": info["suggestions"]
                }
                for dim, info in decision_dict["dimension_scores"].items()
            },
            "next_chapter_creative_direction": decision_dict["next_chapter_creative_direction"]
        }

        return {
            "committee_decision": committee_decision,
            "required_agents": required_agents
        }

    except Exception as e:
        print(f"--- ❌ Error in review_agent: {e} ---")
        return {
            "committee_decision": error_decision,
            "required_agents": ["emotional_reader_agent", "rhythm_reader_agent", "immersion_reader_agent",
                                "structural_novelist_agent", "foreshadowing_novelist_agent"]
        }


def review_agent_test():
    """测试 Review Agent 基本功能"""
    print("🧪 测试 Review Agent 基本功能...")

    # 创建模拟状态 - 正常情况
    test_state: StoryState = {
        "title": "测试小说",
        "story_outline": "这是一个测试故事的大纲",
        "current_chapter_index": 1,
        "chapter_title": "测试章节",
        "chapter_outline": "这是测试章节的大纲",
        "creative_brief": {
            "narrative_goals": ["推进剧情", "发展角色"],
            "character_focus": ["主角A"],
            "thematic_elements": ["成长", "挑战"],
            "structural_requirements": ["三幕式结构"]
        },
        "revised_draft": "这是修订后的章节草稿内容...",
        "expert_evaluations": [
            {
                "agent_id": "emotional_reader_agent",
                "specialization": "情感读者",
                "dimension_scores": {"emotional": 8},
                "evaluation_comments": {"emotional": "情感表达良好"}
            },
            {
                "agent_id": "rhythm_reader_agent",
                "specialization": "节奏读者",
                "dimension_scores": {"rhythm": 7},
                "evaluation_comments": {"rhythm": "节奏把控尚可"}
            }
        ],
        "full_text_history": ["第一章内容"],
        "summary_history": ["第一章摘要"],
        "suggestions": []
    }

    try:
        result = review_agent(test_state)
        print("✅ Review Agent 测试通过!")
        print(f"决策结果: {result.get('committee_decision', {}).get('overall_verdict', 'N/A')}")
        print(f"需要执行的Agent: {result.get('required_agents', [])}")
        return True

    except Exception as e:
        print(f"❌ Review Agent 测试失败: {e}")
        return False


def review_agent_error_test():
    """测试 Review Agent 错误处理"""
    print("🧪 测试 Review Agent 错误处理...")

    # 创建模拟状态 - 缺少必要数据
    test_state: StoryState = {
        "title": "测试小说",
        "current_chapter_index": 1,
        "revised_draft": "",  # 空的修订稿
        "expert_evaluations": [],  # 空的专家评估
        "suggestions": []
    }

    try:
        result = review_agent(test_state)
        print("✅ Review Agent 错误处理测试通过!")
        print(f"默认决策: {result.get('committee_decision', {}).get('overall_verdict', 'N/A')}")
        print(f"默认Agent列表: {result.get('required_agents', [])}")
        return True

    except Exception as e:
        print(f"❌ Review Agent 错误处理测试失败: {e}")
        return False


def run_review_tests():
    """运行所有 Review Agent 测试"""
    print("🚀 开始运行 Review Agent 测试...\n")

    tests_passed = 0
    tests_total = 2

    # 测试正常功能
    if review_agent_test():
        tests_passed += 1
    print()

    # 测试错误处理
    if review_agent_error_test():
        tests_passed += 1
    print()

    # 输出测试结果
    print(f"📊 Review Agent 测试结果: {tests_passed}/{tests_total} 通过")

    if tests_passed == tests_total:
        print("🎉 所有 Review Agent 测试通过!")
    else:
        print("⚠️  部分测试失败，需要检查代码")

    return tests_passed == tests_total


if __name__ == "__main__":
    run_review_tests()