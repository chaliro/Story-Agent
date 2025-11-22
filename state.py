import operator

from pydantic import BaseModel

from memory import MemorySystem

"""
为多Agent协同叙事创作系统定义共享状态的数据结构。

该文件包含主 StoryState TypedDict 以及所有其依赖的组件数据结构。
这些结构经过专门设计，旨在支持一个包含“建议-修改-评审-决策-循环”的先进工作流程。
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated

class GraphEditRequest(BaseModel):
    operation: str  # "delete_node", "delete_edge", "update_node"
    data: Dict[str, Any]  # 包含 name, char_a, char_b, new_profile 等
# 为了在类型提示中使用 MemorySystem 而不产生循环导入问题，
# 我们可以使用前向引用字符串 'MemorySystem'。
# from memory_system import MemorySystem  # 实际使用时会像这样导入
# === 新增 ===
class ConfirmPublishRequest(BaseModel):
    publish_content: str
    tool_calls: List[Dict[str, Any]]  # 前端确认后的操作列表

# ================================================================= #
# 1. 组件数据结构 (Component Data Structures)
# ================================================================= #
class ChapterData(BaseModel):
    input: str
    chapter_title: str
    chapter_outline: str
    creative_brief: Dict[str, Any]

class GenChapterData(BaseModel):
    chapter_title: str
    chapter_outline: str
    creative_brief: Dict[str, Any]
class ProjectCreateRequest(BaseModel):
    project_name: str
    title: str
    story_outline: str
    total_text_style: str
# 定义请求数据模型
class StoryData(BaseModel):
    title: str
    outline: str
    style: str
    prompt: str

# 定义响应数据模型
class publish_state(BaseModel):
    publish_content: str
class publish_content(BaseModel):
    name: str
class AIResponse(BaseModel):
    suggestions: str
    updatedFields: Optional[Dict[str, str]] = None
class Suggestion(TypedDict):
    """
    代表来自“建议型”Agent（如情感、节奏专家）的反馈。
    该结构只包含可操作的修改建议，不包含评分。
    """
    agent_id: str
    specialization: str
    # 示例: [{"location": "第三段", "suggestion": "增加主角的内心挣扎..."}, ...]
    suggestions: List[Dict[str, Any]]

class ExpertEvaluation(TypedDict):
    """
    代表来自“专家评审”Agent（如文学性、结构性专家）的评分和评语。
    该结构用于在修订稿生成后，对其质量进行量化和定性评估。
    """
    agent_id: str
    specialization: str
    score: float  # 范围为 1-5 分
    comments: str

class CommitteeDecision(TypedDict):
    """
    代表来自 review_agent 的最终决策包。
    它综合了专家评审的意见，生成各维度的最终分数，并指导后续流程。
    """
    overall_verdict: str  # 例如: "PUBLISH", "FORCE_PUBLISH", "REVISE"
    # 例如: {"emotion": 4.5, "rhythm": 3.8, ...}
    dimension_scores: Dict[str, Any]

    next_chapter_creative_direction: Dict[str, Any]


# ================================================================= #
# 2. 主状态图定义 (Main State Graph Definition)
# ================================================================= #

class StoryState(TypedDict):
    """
    定义整个多Agent写作工作流的完整共享状态。
    这是所有节点之间信息传递的唯一真实来源（Single Source of Truth）。
    """
    # --- 基础小说信息 ---
    title: str
    story_outline: str

    # --- 当前章节创作信息 ---
    current_chapter_index: int
    chapter_title: str
    chapter_outline: str
    creative_brief: Dict[str, Any]

    agent_flags: Dict[str, int]
    required_agents: List[str]
    # --- 流程中的核心产物 ---
    initial_draft: Optional[str]   # 由 writer_agent 生成的初稿
    revised_draft: Optional[str]   # 由 rewriter_agent 生成的修订稿

    # --- 循环与反馈管理 ---
    rewrite_attempts: int          # 用于控制重写循环的计数器（最多5次）
    suggestions: Annotated[List[Suggestion], operator.add]  # 存储来自五个“建议型”Agent的反馈
    expert_evaluations: List[ExpertEvaluation] # 存储来自三个“专家评审”Agent的评估

     # +++ 新增：存储每次修订的草稿及其评分 +++
    chapter_versions: List[Dict[str, Any]]

    # +++ 新增：存储每次修订的草稿及其评分 +++
    total_text_style:str

    # +++ 新增：存储给重写循环的针对性指令 +++
    revision_brief: Optional[Dict[str, Any]]
    # --- 最终决策 ---
    committee_decision: Optional[CommitteeDecision]

    # --- 系统与历史记录 ---
    full_text_history: List[str]   # 存储已发布章节的完整文本
    chapter_history:List[str]
    summary_history: List[str]     # 存储已发布章节的摘要
    final_chapter: Optional[str]   # 存储最终发布的章节内容
    # +++ 修改：存储最终发布的章节内容 +++
    published_chapter: Optional[str]