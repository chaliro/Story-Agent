import json
import os
from typing import Optional, Dict, Tuple

import dotenv

from helper import get_memory_path
from memory import MemorySystem
from utils import get_embedding_llm


def initialize_projects_file():
    """初始化项目管理文件"""
    dotenv.load_dotenv()
    PROJECTS_FILE = os.getenv("PROJECTS_FILE")
    if not os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump({"next_id": 1, "projects": {}}, f, ensure_ascii=False, indent=2)
        print("✅ 已初始化项目管理文件")


def get_projects() -> Dict:
    """获取所有项目信息"""
    initialize_projects_file()
    dotenv.load_dotenv()
    PROJECTS_FILE = os.getenv("PROJECTS_FILE")
    with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_projects(data: Dict):
    """保存项目信息"""
    dotenv.load_dotenv()
    PROJECTS_FILE = os.getenv("PROJECTS_FILE")
    with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_project(project_name: str, title: str, story_outline: str, total_text_style: str) -> Tuple[bool, str, Optional[int]]:
    """
    创建新项目
    返回：(是否成功, 消息, 项目ID)
    """
    projects_data = get_projects()

    # 检查项目名称是否已存在
    for pid, pinfo in projects_data["projects"].items():
        if pinfo["name"] == project_name:
            return (False, f"项目 '{project_name}' 已存在", None)

    # 创建新项目
    project_id = projects_data["next_id"]
    dotenv.load_dotenv()
    PROJECTS_FILE = os.getenv("PROJECTS_FILE")
    projects_data["projects"][str(project_id)] = {
        "name": project_name,
        "created_at": os.path.getctime(PROJECTS_FILE)  # 简化的创建时间
    }
    projects_data["next_id"] += 1
    save_projects(projects_data)
    dotenv.load_dotenv()
    # 创建项目目录
    MEMORY_ROOT = os.getenv("MEMORY_ROOT")
    project_dir = os.path.join(MEMORY_ROOT, str(project_id))
    os.makedirs(project_dir, exist_ok=True)
    embeddings = get_embedding_llm()
    memory_dir = project_dir

    memory_system = MemorySystem()
    memory_system.save(memory_dir)

    initial_state = {
        "title": title,
        "story_outline": story_outline,
        "total_text_style": total_text_style,
        "current_chapter_index": 1,
        "full_text_history": [],
        "summary_history": [],
        "committee_decision": None,
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
    # 初始化项目内的状态文件和知识库文件
    with open(os.path.join(project_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(initial_state, f, ensure_ascii=False, indent=2)

    with open(os.path.join(project_dir, "knowledge_base.json"), "w", encoding="utf-8") as f:
        json.dump({"characters": {}, "last_updated_chapter": 0}, f, ensure_ascii=False, indent=2)
    chapter_path = os.path.join(project_dir, "chapter.json")
    if os.path.exists(chapter_path):
        with open(chapter_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # 文件不存在，创建并初始化
        data = {'current_index': 1}
    # 保存回文件
    with open(chapter_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



    return (True, f"项目 '{project_name}' 创建成功，ID: {project_id}", project_id)


def get_project_path(project_id: int) -> Optional[str]:
    """获取项目的存储路径"""
    dotenv.load_dotenv()
    MEMORY_ROOT = os.getenv("MEMORY_ROOT")
    projects_data = get_projects()
    if str(project_id) not in projects_data["projects"]:
        return None
    return os.path.join(MEMORY_ROOT, str(project_id))


def get_project_id_by_name(name: str) -> Optional[int]:
    """通过项目名称获取ID"""
    projects_data = get_projects()
    for pid, pinfo in projects_data["projects"].items():
        if pinfo["name"] == name:
            return int(pid)
    return None

if __name__ == "__main__":
# 创建项目
    success, msg, pid = create_project("我的科幻小说")
    print(msg)  # 项目 '我的科幻小说' 创建成功，ID: 1

# 查找项目ID
    project_id = get_project_id_by_name("我的科幻小说")
    print(project_id)  # 1

# 获取项目路径
    project_path = get_project_path(project_id)
    print(project_path)  # ./memory/1

# 访问项目内的文件
    state_path = os.path.join(project_path, "state.json")
    knowledge_path = os.path.join(project_path, "knowledge_base.json")