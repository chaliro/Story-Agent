import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from state import StoryState, ExpertEvaluation
from utils import get_llm, get_evaluation_llm
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from helper import filter_think_tags, parse_json_with_filtering


# ==================== 定义 Pydantic 模型 ====================

class DimensionScore(BaseModel):
    score: int = Field(description="维度评分，范围1-10")
    comment: str = Field(description="维度评语")


class NarrativeExpertEvaluation(BaseModel):
    agent_id: str = Field(default="Narrative_Expert", description="代理ID")
    specialization: str = Field(default="叙事结构专家", description="专业领域")
    dimension_scores: Dict[str, int] = Field(description="维度评分")
    evaluation_comments: Dict[str, str] = Field(description="维度评语")


class ReaderExperienceEvaluation(BaseModel):
    agent_id: str = Field(default="Reader_Experience_Expert", description="代理ID")
    specialization: str = Field(default="读者体验专家", description="专业领域")
    dimension_scores: Dict[str, int] = Field(description="维度评分")
    evaluation_comments: Dict[str, str] = Field(description="维度评语")


class LiteraryArtEvaluation(BaseModel):
    agent_id: str = Field(default="Literary_Art_Expert", description="代理ID")
    specialization: str = Field(default="文学艺术专家", description="专业领域")
    dimension_scores: Dict[str, int] = Field(description="维度评分")
    evaluation_comments: Dict[str, str] = Field(description="维度评语")


# 专家配置



def expert_evaluation_agent(state: StoryState) -> Dict[str, Any]:
    print("--- 🌟 专家评分系统：开始对改写内容进行多维度评估... ---")
    experts_config = [
        {
            "id": "Narrative_Expert",
            "specialization": "叙事结构专家",
            "focus": "擅长整体叙事架构和情节推进逻辑评估",
            "evaluation_focus": "从叙事系统整体性出发，评估故事结构的完整性和逻辑自洽性",
            "model_class": NarrativeExpertEvaluation,
            "dimensions": [
                {
                    "id": "worldview_integration",
                    "name": "世界观-情节融合度",
                    "description": "评估世界观设定与情节发展的深度整合，科技/魔法体系是否有机影响故事走向",
                    "score_examples": [
                        "1-3分：世界观与情节完全脱节，设定仅为装饰性背景",
                        "4-6分：世界观能支撑基本情节但缺乏深度互动",
                        "7-10分：世界观规则深度驱动情节发展，形成有机整体"
                    ],
                    "comment_examples": [
                        "优点：图书馆的腐朽细节与废弃原因形成呼应；不足：挂坠盒的能量来源未充分融入世界观体系"
                    ]
                },
                {
                    "id": "narrative_coherence",
                    "name": "叙事逻辑连贯性",
                    "description": "评估情节推进的逻辑合理性，伏笔与解谜的呼应关系",
                    "score_examples": [
                        "1-3分：情节发展缺乏逻辑基础，行为动机不明确",
                        "4-6分：基本逻辑合理但存在牵强之处",
                        "7-10分：情节发展自然流畅，前后呼应形成完整逻辑链"
                    ],
                    "comment_examples": [
                        "优点：祖父日记作为线索贯穿始终；不足：妹妹回忆与当前探索的关联逻辑需加强"
                    ]
                }
            ]
        },
        {
            "id": "Reader_Experience_Expert",
            "specialization": "读者体验专家",
            "focus": "专注于读者代入感、情感共鸣和阅读舒适度评估",
            "evaluation_focus": "从普通读者视角评估内容的可读性、情感冲击力和注意力保持效果",
            "model_class": ReaderExperienceEvaluation,
            "dimensions": [
                {
                    "id": "emotional_accessibility",
                    "name": "情感共鸣易达性",
                    "description": "评估情感描写是否能让不同背景读者快速产生共鸣和代入感",
                    "score_examples": [
                        "1-3分：情感表达抽象晦涩，难以引发读者共鸣",
                        "4-6分：情感合理但缺乏具象细节支撑",
                        "7-10分：通过具体行为和细节自然传递情感，读者能轻松代入"
                    ],
                    "comment_examples": [
                        "优点：妹妹笑容的回忆用'驱散阴霾'建立情感连接；不足：恐惧情绪描写可增加更多生理反应细节"
                    ]
                },
                {
                    "id": "sensory_immersion",
                    "name": "多感官沉浸体验",
                    "description": "评估感官细节的丰富度和层次感，能否让读者快速构建生动场景想象",
                    "score_examples": [
                        "1-3分：感官描写单一或缺失，场景想象困难",
                        "4-6分：有基本感官描写但缺乏层次和细节",
                        "7-10分：多感官联动形成立体体验，场景栩栩如生"
                    ],
                    "comment_examples": [
                        "优点：挂坠盒的温度变化提供明确触觉锚点；不足：环境声音和气味细节可进一步丰富"
                    ]
                }
            ]
        },
        {
            "id": "Literary_Art_Expert",
            "specialization": "文学艺术专家",
            "focus": "擅长从文学技巧和艺术表达角度评估内容质量",
            "evaluation_focus": "从文学创作角度评估表达的艺术性、创新性和美学价值",
            "model_class": LiteraryArtEvaluation,
            "dimensions": [
                {
                    "id": "foreshadowing_artistry",
                    "name": "伏笔艺术表现力",
                    "description": "评估伏笔设置的巧妙程度，是否通过隐喻/象征手法呈现多义解读空间",
                    "score_examples": [
                        "1-3分：伏笔直白生硬，缺乏艺术处理",
                        "4-6分：有基本伏笔但象征单一，解读空间有限",
                        "7-10分：伏笔巧妙融入叙事，具备多层次象征意义"
                    ],
                    "comment_examples": [
                        "优点：月光'如同时光流逝'既写景又暗示时间线索；不足：锁链声响的象征意义可进一步挖掘"
                    ]
                },
                {
                    "id": "language_rhythm",
                    "name": "语言节奏艺术性",
                    "description": "评估句式结构、标点运用对叙事节奏的调控效果",
                    "score_examples": [
                        "1-3分：句式单调平板，缺乏节奏变化",
                        "4-6分：有节奏变化但与情节情绪匹配度不高",
                        "7-10分：语言节奏精准匹配情节起伏，增强叙事张力"
                    ],
                    "comment_examples": [
                        "优点：环境描写用长句营造沉浸感，突发声响用短句制造冲击；不足：回忆与现实间的节奏过渡可优化"
                    ]
                },
                {
                    "id": "structural_innovation",
                    "name": "叙事结构创新性",
                    "description": "评估时空转换、视角切换等结构手法的艺术运用效果",
                    "score_examples": [
                        "1-3分：结构传统单一，缺乏创新表达",
                        "4-6分：有结构变化但手法常规",
                        "7-10分：结构手法创新且服务于主题表达，增强艺术感染力"
                    ],
                    "comment_examples": [
                        "优点：用挂坠盒作为时空转换锚点；不足：章节结尾悬念设置的艺术性可加强"
                    ]
                }
            ]
        }
    ]
    revised_draft = state.get('revised_draft')
    if not revised_draft:
        print("--- ⚠️ 警告：未找到改写后的草稿内容 ---")
        return {"expert_evaluations": []}

    state["expert_evaluations"] = None
    all_evaluations: List[ExpertEvaluation] = []

    for expert in experts_config:
        print(f"--- 👤 {expert['specialization']} 正在评分... ---")

        # 创建输出解析器
        parser = PydanticOutputParser(pydantic_object=expert["model_class"])

        # 生成专家专属维度描述
        dimension_descriptions = "\n".join([
            f"- {dim['id']}: {dim['name']}\n"
            f"描述：{dim['description']}\n"
            f"评分参考：{'; '.join(dim['score_examples'])}\n"
            f"评语示例：{dim['comment_examples'][0]}"
            for dim in expert["dimensions"]
        ])

        # 构建提示词模板
        prompt_template = """
你是一名资深的{specialization}，专注于{focus}，核心评估准则：{evaluation_focus}。
请基于以下内容，仅从你专精的维度进行评分（其他维度无需关注）：

**改写后章节内容**:
{revised_draft}

**你的专属评分维度及说明**:
{dimension_descriptions}

**评分要求**:
1. 每个维度评分范围为1-10分，严格参考你的专属评分示例
2. 评语需结合你的专业视角（如读者体验/文学技巧）
3. 必须严格按照指定JSON格式输出，不要添加任何额外内容

**输出要求**:
请严格按照指定的JSON格式提供评估报告。

{format_instructions}
"""

        llm = get_evaluation_llm()

        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["specialization", "focus", "evaluation_focus", "revised_draft",
                                 "dimension_descriptions"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            chain = prompt | llm
            raw_response = chain.invoke({
                "specialization": expert["specialization"],
                "focus": expert["focus"],
                "evaluation_focus": expert["evaluation_focus"],
                "revised_draft": revised_draft,
                "dimension_descriptions": dimension_descriptions
            })

            # 过滤思考标签并解析
            filtered_content = filter_think_tags(raw_response.content)
            parsed_evaluation = parse_json_with_filtering(filtered_content, parser)

            all_evaluations.append(parsed_evaluation.model_dump())
            print(f"--- ✅ {expert['specialization']} 评分完成 ---")

        except Exception as e:
            print(f"--- ❌ {expert['specialization']} 评分失败: {e} ---")
            # 添加错误评估作为占位符
            error_evaluation = {
                "agent_id": expert["id"],
                "specialization": expert["specialization"],
                "dimension_scores": {dim["id"]: 0 for dim in expert["dimensions"]},
                "evaluation_comments": {dim["id"]: f"评分失败: {str(e)}" for dim in expert["dimensions"]}
            }
            all_evaluations.append(error_evaluation)

    print("--- 🏁 所有专家评分完成 ---")
    return {"expert_evaluations": all_evaluations}


# ==================== 测试函数 ====================

def expert_evaluation_agent0():
    """测试专家评估 Agent"""
    print("🧪 测试专家评估 Agent...")

    # 创建模拟状态
    test_state: StoryState = {
        "revised_draft": "这是一个测试章节内容。主角在废弃的图书馆中发现了一个神秘的挂坠盒，挂坠盒散发着微弱的光芒。当他触摸挂坠盒时，脑海中浮现出妹妹的笑容，这让他下定决心要继续探索下去。",
        "expert_evaluations": None,
        "agent_flags": {}
    }

    result = expert_evaluation_agent(test_state)
    print("专家评估 Agent 测试结果:")
    # print(f"评估数量: {len(result.get('expert_evaluations', []))}")
    #
    # for evaluation in result.get('expert_evaluations', []):
    #     print(f"专家: {evaluation.get('specialization', '未知')}")
    #     print(f"维度评分: {evaluation.get('dimension_scores', {})}")
    #     print("---")
    print(result)

    return True


def expert_evaluation_empty_draft0():
    """测试专家评估 Agent 无草稿情况"""
    print("🧪 测试专家评估 Agent 无草稿情况...")

    # 创建模拟状态 - 没有草稿
    test_state: StoryState = {
        "revised_draft": None,
        "expert_evaluations": None,
        "agent_flags": {}
    }

    result = expert_evaluation_agent(test_state)
    print("专家评估 Agent 无草稿测试结果:")
    print(f"评估列表为空: {result.get('expert_evaluations') == []}")
    return True


def run_all_evaluation_tests():
    """运行所有 Evaluation Agents 测试"""
    print("🚀 开始测试 Evaluation Agents...\n")

    expert_evaluation_agent0()
    expert_evaluation_empty_draft0()

    return


# 在文件末尾添加以下代码来运行测试
if __name__ == "__main__":
    # 如果直接运行这个文件，执行所有测试
    run_all_evaluation_tests()