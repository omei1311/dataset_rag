from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app
from app.utils.query_runs import QueryInProgressError, SessionRunRegistry

_query_runs = SessionRunRegistry()


# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康状态
@app.get("/health")
async def health():
    logger.info(f"触发后台检测检查接口，数据一切正常！！")
    return {"status": "ok"}


# 返回chat.html
@app.get("/chat.html")
async def chat_html():
    # 查找chat.html页面的地址
    chat_html_path  = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html文件不存在")
    return FileResponse(chat_html_path)


# 发起提问接口
# 接收参数的类型
class QueryRequest(BaseModel):
    query: str = Field(..., title="查询内容,必须传递")
    session_id:str = Field(None, title="会话id，可以不传递，后台uuid生成第一个！")
    is_stream:bool = Field(False, title="是否流式返回结果")


def run_query_graph(query: str, session_id: str, is_stream: bool,
                    *, already_claimed: bool = False, release_on_finish: bool = True):
    if not already_claimed and not _query_runs.acquire(session_id):
        raise QueryInProgressError("同一会话已有查询正在执行")
    try:
        # Only the owning run may reset its in-memory state. MongoDB history is
        # separate and intentionally retained for multi-turn conversations.
        clear_task(session_id)
        update_task_status(session_id, "processing", is_stream)
        state = create_query_default_state(
            session_id=session_id, original_query=query, is_stream=is_stream
        )
        query_app.invoke(state)
        # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
        update_task_status(session_id, "completed", is_stream)
        return True
    except Exception as e:
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        # Discard partial output as well as previous answers. A failed query must
        # never look like a successful response with stale evidence.
        clear_task(session_id)
        update_task_status(session_id, "failed", is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": "本次查询处理失败，请稍后重试"})
        return False
    finally:
        if release_on_finish:
            _query_runs.release(session_id)


@app.post("/query")  # 客户端 -》 问题 -》 graph开启了 -》 查到rag的结果 -》 返回即可！！
async def query(request: QueryRequest,background_tasks: BackgroundTasks):
    """
    :param request: 请求参数
    :param background_tasks: 异步执行函数  is_stream = True
    :return:
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    if not _query_runs.acquire(session_id):
        raise HTTPException(status_code=409, detail="同一会话已有查询正在执行，请等待完成后重试")
    # 判断是不是流式处理 （异步 -》 先返回一个结果 开始处理 | 后台运行图，结果向前端推送）
    if is_stream:
        # 只要开启流式处理，我们业务中就是将数据，插入到队列中！ {session_id , queue [update_task_state , add_running_task,add_done_list]}
        # 创建当前session_id对应的队列 =》 _session_stream
        try:
            clear_task(session_id)
            update_task_status(session_id, "pending")
            create_sse_queue(session_id)
            background_tasks.add_task(
                run_query_graph, query, session_id, is_stream, already_claimed=True
            )
            return {"session_id": session_id, "message": "本次查询处理中...."}
        except Exception:
            _query_runs.release(session_id)
            raise
    else:
        # 同步执行
        try:
            ok = run_query_graph(query, session_id, is_stream,
                                 already_claimed=True, release_on_finish=False)
            if not ok:
                raise HTTPException(status_code=500, detail="本次查询处理失败，请稍后重试")
            return {
                "answer": get_task_result(session_id, "answer"),
                "sources": get_task_result(session_id, "sources", []),
                "rerank_degraded": bool(get_task_result(session_id, "rerank_degraded", False)),
                "session_id": session_id,
                "message": "本次查询处理完毕！",
                "done_list": []
            }
        finally:
            _query_runs.release(session_id)


@app.get("/stream/{session_id}")
async def stream(session_id: str,request:Request):
    """
    :param session_id:
    :param request: 前端的原生请求对象，可以判断是否断开连接！！
    :return:
    """
    logger.info(f"session_id = {session_id}客户端，已经和后台建立了长连接！")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream"
    )


@app.get("/history/{session_id}")
async def history(session_id: str,limit: int = 10):
    """
    :param session_id:
    :param limit: 切割的数量
    :return:
    """
    # 获取历史对话
    chats =  get_recent_messages(session_id,limit)
    # items = []
    # for chat in chats:
    #     items.append(chat)
    logger.info(f"查询历史对话，session_id = {session_id}成功！查询数据为：{chats}")
    return {
        "session_id":session_id,
        "items": chats
    }

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    :param session_id:
    :return:
    """
    # 删除历史对话
    delete_count = clear_history(session_id)
    logger.info(f"删除历史对话，session_id = {session_id}成功,删除数量：{delete_count}！")
    return {
        "deleted_count":delete_count,
        "message": f"{session_id}聊天记录删除成功！"
    }


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)
