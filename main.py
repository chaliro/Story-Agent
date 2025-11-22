import json
import pprint
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dotenv
from dotenv import set_key
from fastapi import FastAPI, Request, Form, HTTPException, Body, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os

from character import initialize_knowledge_base, run_complete_relationship_analysis, \
    simulate_user_confirmation_and_execute, get_knowledge_base
from graph import get_workflow
from helper import get_ai_suggestions, planned, update_json_with_dict, get_memory_path, publish_chapter, \
    finalize_chapter_and_save_state, put_chapter, get_all_chapter
from project_manager import create_project, get_projects  # 导入项目管理模块
from state import AIResponse, StoryData, ProjectCreateRequest, ChapterData, GenChapterData, publish_state, \
    publish_content
from utils import get_llm
import dotenv
# 初始化FastAPI应用
app = FastAPI(title="AI小说创作辅助系统")

# 配置模板目录（指向scripts文件夹）
templates = Jinja2Templates(directory="scripts")


# 首页路由 - 提供HTML界面
@app.get("/index.html", response_class=HTMLResponse)
async def read_index(request: Request):
    """返回首页HTML页面"""
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/api/projects", response_class=JSONResponse)
async def create_new_project(project_data: ProjectCreateRequest):
    """
    创建新的小说项目
    接收 JSON 格式的请求体，例如: {"project_name": "我的新小说"}
    """
    # FastAPI 会自动解析请求体的 JSON，并将其转换为 ProjectCreateRequest 对象
    # 同时，如果 JSON 格式错误或缺少 project_name 字段，会自动返回 422 Unprocessable Entity 错误

    project_name = project_data.project_name.strip()
    title = project_data.title.strip()
    story_outline = project_data.story_outline.strip()
    total_text_style = project_data.total_text_style.strip()

    if not project_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="项目名称不能为空或仅包含空格"
        )

    # 调用项目管理模块创建项目
    success, message, project_id = create_project(project_name,title,story_outline,total_text_style)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    return
    return {
        "success": success,
        "message": message,
        "project_id": project_id
    }
# 在 main.py 中添加
@app.get("/project_setup.html", response_class=HTMLResponse)
async def project_setup(request: Request):
    return templates.TemplateResponse("project_setup.html", {"request": request})

@app.get("/write.html", response_class=HTMLResponse)
async def project_setup(request: Request, project_id: int):
    print("==================")
    print(project_id)
    dotenv.load_dotenv()
    set_key(".env", "CURRENT_PROJECT_ID", str(project_id))
    os.environ["CURRENT_PROJECT_ID"] = str(project_id)  # 直接更新环境变量
    dotenv.load_dotenv()

    return templates.TemplateResponse("write.html", {"request": request})

# 项目列表API（可选，用于查看所有项目）
@app.get("/api/projects", response_class=JSONResponse)
async def list_projects():
    """获取所有项目信息"""
    projects_data = get_projects()
    return {
        "total": len(projects_data["projects"]),
        "projects": projects_data["projects"]
    }
@app.post("/api/chapterIntroduction", response_class=JSONResponse)
async def chapterIntroduction(chapter: ChapterData):
    """获取所有项目信息"""
    print(chapter)
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")
    dir3 = "state.json"
    input = chapter.input
    chapter_title = chapter.chapter_title
    chapter_outline = chapter.chapter_outline
    creative_brief = chapter.creative_brief
    chapter_data = {
        "chapter_title": chapter_title,
        "chapter_outline": chapter_outline,
        "creative_brief": creative_brief
    }
    # chapter_json = json.dumps(chapter_data, indent=4, ensure_ascii=False)
    # print("==========")

    dir = os.path.join(dir1, dir2, dir3)
    dotenv.load_dotenv()
    update_json_with_dict(dir,chapter_data)
    memory_path = os.path.join(dir1,dir2)
    planned(input, dir, memory_path)
    with open(dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 数据
    response_data = {
        "chapter_title": data["chapter_title"],
        "chapter_outline": data["chapter_outline"],
        "creative_brief": data["creative_brief"]
    }
    return response_data
@app.post("/api/write", response_class=JSONResponse)
async def chapterWrite(chapter: GenChapterData):
    """获取所有项目信息"""
    print(chapter)
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")

    dir3 = "state.json"

    chapter_title = chapter.chapter_title
    chapter_outline = chapter.chapter_outline
    creative_brief = chapter.creative_brief
    chapter_data = {
        "chapter_title": chapter_title,
        "chapter_outline": chapter_outline,
        "creative_brief": creative_brief
    }

    print(type(chapter_data))
    dir = os.path.join(dir1, dir2, dir3)
    update_json_with_dict(dir,chapter_data)

    with open(dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 数据
    workflow = get_workflow()
    final_state = workflow.invoke(data)
    #st = """梦境的回响 流星雨如同夜空的泪水，悄然滑落，映照出小雨内心深处的渴望与无助。她的眼睑渐渐沉重，仿佛那一颗颗流星不仅划过天空，更拉扯着她的灵魂，引她走向一个未知的梦境。意识在星辰的光华中游荡，缓缓滑入那片无垠的星空，那里是她内心欲望与失落交织成的奇幻国度。 星空璀璨，流动的光影轻轻舞动，似无数记忆的碎片在宇宙中闪烁。小雨站在一片梦幻的星海中，耳边回荡着温柔的低语，那是她熟悉却又遥远的声音。转过身来，眼前的一幕令她心中一震——父母微笑着向她走来，脸上的温暖如阳光洒落在她的心田。 “妈妈，爸爸！”小雨的声音在梦中显得脆弱，却又满含期盼。 他们的微笑宛如晨曦中的露珠，清澈而动人。母亲轻轻抚摸着小雨的头发，父亲用温暖的怀抱将她包围，那个久违的感觉如潮水般涌来，仿佛她的心灵终于找到了归宿。然而，在这片刻的安宁中，小雨的内心却悄然泛起一丝不安。她知道，这一切不过是心中深埋的回忆，无法触及的幻影。 “我们一直在你身边。”母亲的声音如同轻柔的风，抚慰着小雨的心灵。 小雨抬头，目光穿透星空的迷雾，似乎看见了那些温暖的瞬间：她与父母在阳光下奔跑的画面，欢笑声在耳边回响，时光如流星划过，短暂却璀璨。她想与他们分享自己这些年的经历，分享对未来的迷茫与追求。 “我不知道我该怎么做。”小雨声音低沉而无助，眼泪在眼眶中打转，“我总是感到迷失，仿佛被一种无形的力量束缚着。” “失去是我们共同的痛苦，但它也是你成长的一部分。”父亲的话如星光，透过梦境的迷雾，直抵小雨的心底。他的语调温柔而坚定，仿佛每一个字都在撼动她的灵魂。 小雨的心中涌起一阵暖流，父母的存在不仅是她内心对温暖的渴望，更是对她勇气的呼唤。随着父亲的话语，小雨的思绪逐渐明晰，那些美好的回忆虽然温暖，却也成为了她前行的枷锁。她必须放下这段过去，才能真正迎接未来。 在星空的深处，流星划过，留下一道耀眼的光芒。小雨不再只是静静地仰望，而是开始追逐那道光，心中涌动着从未有过的力量。她奔跑着，追逐着那闪烁的流星，感受着心跳的激昂与生命的澎湃。 最终，梦境中的父母微笑着向她挥手，宛如晨曦中的星辰，渐渐远去。小雨知道，她必须告别这段过往，才能在现实中找到自己的方向。梦的深处，她感受到一种前所未有的勇气与坚定。就在这时，流星雨的光辉洒落窗外，像是对她未来旅程的祝福。 小雨的心灵如同那璀璨的星空，充满希望与勇气，准备迎接属于她的明天。在梦中下定决心的那一刻，她明白了，生活的挑战在前方等待，而她将不再逃避自己的内心，而是勇敢地迎接即将到来的每一个选择与改变。"""
    # #print(final_state["title"])

    return {"chapter": final_state["final_chapter"]}
    #return {"chapter": st}
@app.post("/api/publish", response_class=JSONResponse)
async def chapterPublish(state: publish_state):
    """获取所有项目信息"""
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")
    dir4 = "chapter.json"
    dir3 = "state.json"
    dir5 = "knowledge_base.json"

    dir = os.path.join(dir1, dir2, dir3)
    chapter_path = os.path.join(dir1, dir2, dir4)
    memory_path = get_memory_path()
    knowledge_path = os.path.join(dir1, dir2, dir5)
    best_draft = state.publish_content
    with open(dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 数据
    final_state = publish_chapter(data, best_draft)
    put_chapter(final_state["chapter_title"], final_state["published_chapter"], chapter_path)
    # 1. 初始化知识库
    initialize_knowledge_base(knowledge_path)


    tool_calls_ch1 = run_complete_relationship_analysis(best_draft, current_chapter=final_state["current_chapter_index"] - 1)
    simulate_user_confirmation_and_execute(tool_calls_ch1)

    final_kb = get_knowledge_base(knowledge_path)
    pprint.pprint(final_kb)
    finalize_chapter_and_save_state(final_state, dir, memory_path)
    response = get_all_chapter(chapter_path).keys()
    res = list(response)
    res = res.remove("current_index")
    return {"response": res}
# 新增接口：获取书籍目录
@app.get("/api/book_catalog", response_class=JSONResponse)
async def get_book_catalog(project_id: str):
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = project_id  # 使用传入的 project_id
    dir4 = "chapter.json"

    chapter_path = os.path.join(dir1, dir2, dir4)
    response = get_all_chapter(chapter_path).keys()
    res = list(response)
    if "current_index" in res:
        res.remove("current_index")
    return {"response": res}
@app.post("/api/get_chapter", response_class=JSONResponse)
async def chapterPublish(state: publish_content):
    """获取所有项目信息"""
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")
    dir4 = "chapter.json"
    chapter_path = os.path.join(dir1, dir2, dir4)
    name = state.name
    with open(chapter_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 数据

    return {"content": data.get(name, "章节内容不存在")}
# 添加获取AI建议的API
@app.post("/api/ai/suggest", response_class=JSONResponse)  # 注意路径与前端一致
async def generate_ai_suggestions(
    story_data: StoryData
):
    """
       处理前端AI建议请求
       """
    try:
        # 将结构化数据拼接为字符串，作为LLM的输入
        user_input = _build_user_input(story_data)
        suggestions = get_ai_suggestions(user_input)
        formatted_suggestions = {
            "suggestions": {
                "title": suggestions.get("title", ""),
                "story_outline": suggestions.get("story_outline", ""),
                "total_text_style": suggestions.get("total_text_style", "")
            }
        }
        return formatted_suggestions
    except Exception as e:
        print(f"处理AI建议请求时出错: {e}")
        raise HTTPException(status_code=500, detail="AI服务暂时不可用，请稍后重试")


def _build_user_input(story_data: StoryData) -> str:
    """
    将前端的结构化数据拼接为适合LLM处理的字符串
    """
    user_input_parts = []

    # 添加用户提供的prompt（主要需求）
    if story_data.prompt.strip():
        user_input_parts.append(f"主要需求: {story_data.prompt}")

    # 添加现有标题（如果有）
    if story_data.title.strip():
        user_input_parts.append(f"现有标题: {story_data.title}")

    # 添加现有大纲（如果有）
    if story_data.outline.strip():
        user_input_parts.append(f"现有大纲: {story_data.outline}")

    # 添加现有风格（如果有）
    if story_data.style.strip():
        user_input_parts.append(f"现有风格: {story_data.style}")

    # 如果所有字段都为空，使用prompt作为主要输入
    if not user_input_parts and story_data.prompt.strip():
        return story_data.prompt

    return "\n".join(user_input_parts)


# ==========================================
# 新增：知识图谱可视化接口
# ==========================================
@app.get("/api/graph_data", response_class=JSONResponse)
async def get_graph_data(project_id: str):
    """读取并返回 NetworkX 图谱数据"""
    dotenv.load_dotenv()
    memory_root = os.getenv("MEMORY_ROOT", "./vector_memory")

    # 构造路径：vector_memory/项目ID/story_graph.json
    graph_path = os.path.join(memory_root, str(project_id), "story_graph.json")

    if os.path.exists(graph_path):
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data  # 直接返回 node-link 格式的 JSON
        except Exception as e:
            print(f"读取图谱失败: {e}")
            return {"nodes": [], "links": []}
    else:
        # 如果文件不存在，返回空图
        return {"nodes": [], "links": []}


from state import ConfirmPublishRequest
from character import execute_tool_calls  # 确保导入了这个函数


# ==========================================
# 1. 阶段一：预发布分析 (提取断言与关系)
# ==========================================
@app.post("/api/publish/analyze", response_class=JSONResponse)
async def analyze_publish(state: publish_state):
    """
    预发布：不保存，仅分析文本，提取图谱更新建议，返回给前端确认。
    """
    print("--- 🚀 阶段一：预发布分析 ---")
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")
    dir3 = "state.json"

    state_path = os.path.join(dir1, dir2, dir3)
    best_draft = state.publish_content

    # 读取当前章节索引（不修改文件）
    with open(state_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    current_chapter_index = data.get("current_chapter_index", 1)

    # 调用 character.py 中的分析函数
    # 注意：这里我们不执行，只获取 tool_calls 列表
    print(f"--- 正在分析第 {current_chapter_index} 章的实体与关系 ---")
    tool_calls = run_complete_relationship_analysis(best_draft, current_chapter=current_chapter_index)

    return {
        "tool_calls": tool_calls,
        "message": f"分析完成，生成了 {len(tool_calls)} 条图谱更新建议"
    }


# ==========================================
# 2. 阶段二：确认发布 (执行保存与更新)
# ==========================================
@app.post("/api/publish/confirm", response_class=JSONResponse)
async def confirm_publish(request: ConfirmPublishRequest):
    """
    最终发布：接收用户确认的文本和图谱操作，执行真正的写入。
    """
    print("--- 🚀 阶段二：用户确认发布 ---")
    dotenv.load_dotenv()
    dir1 = os.getenv("MEMORY_ROOT")
    dir2 = os.getenv("CURRENT_PROJECT_ID")

    # 路径定义
    state_path = os.path.join(dir1, dir2, "state.json")
    chapter_path = os.path.join(dir1, dir2, "chapter.json")
    memory_path = get_memory_path()

    # 1. 执行文本发布 (保存章节、更新摘要)
    with open(state_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 使用 helper 中的 publish_chapter 更新内存状态
    final_state = publish_chapter(data, request.publish_content)

    # 写入 chapter.json
    put_chapter(final_state["chapter_title"], final_state["published_chapter"], chapter_path)

    # 2. 【核心】执行用户确认的图谱更新操作 (GraphRAG)
    print(f"--- 正在执行 {len(request.tool_calls)} 条图谱更新操作 ---")
    execute_tool_calls(request.tool_calls)

    # 3. 状态归档 (清理草稿，准备下一章)
    finalize_chapter_and_save_state(final_state, state_path, memory_path)

    # 4. 获取最新的目录返回给前端
    response = get_all_chapter(chapter_path).keys()
    res = list(response)
    if "current_index" in res:
        res.remove("current_index")

    return {"response": res}

if __name__ == "__main__":
    import uvicorn

    # 确保vector_memory目录存在
    if not os.path.exists("./vector_memory"):
        os.makedirs("./vector_memory", exist_ok=True)
    # 启动服务，默认端口8000
    uvicorn.run(app, host="0.0.0.0", port=8000)