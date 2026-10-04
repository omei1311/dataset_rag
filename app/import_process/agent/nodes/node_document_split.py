import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple

from Crypto.SelfTest.Cipher.test_CFB import file_name
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from modelscope.models.cv.video_multi_object_tracking.tracker.multitracker import sub_stracks

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500
"""
   完成md内容的切块！ 
   最终： chunks -> 存储块的集合   chunks ->  备份到本地 -> chunks.json 
   1. 参数校验 
   2. 粗粒度切割（md）语义完善 -》 使用标题切割 
   3. 特殊场景，一个文档没有标题，我们给他一个默认标题
   4. 细粒度切割（md）大小和重叠合适 -> 大 -》（设置重叠） 小 || 小 -》 合并  
      大小合适，语义完整的chunks 
   5. 数据的备份和chunks属性的修改
   返回 state 
"""


def step_1_get_content(state):
    #读取要切片的内容
    md_content = state["md_content"]
    if md_content is None:
        logger.error(f"[step_1_get_content]没有有效的md内容，直接抛出异常！！")
        raise Exception("请检查输入文件路径是否正确")
    md_content = md_content.replace('\r\n','\n').replace('\r','\n')
    file_title = state.get("file_title","default_file")
    return md_content,file_title


def step2_split_by_title(md_content, file_title):
    """
    语义切割
    根据标题进行切割！
    :param md_content: md_content
    :param file_title: file_title
    :return: [{content,title,file_title}]
    """
    # 1. 准备前置工作
    # 1.1 正则
    # \s* 空格 tab 【】######
    title_pattern = r'^\s*#{1,6}\s+.+'

    # 1.2 md_content切割 \n
    lines = md_content.split('\n')

    # 1.3 定义临时存储变量 current_title = str | current_lines = [] | title_count = 0 存储了多少块
    # is_code_block = bool False  是不是代码块
    current_title = ""
    current_lines = []
    title_count = 0
    is_code_block = False

    # 1.4 最终存储的列表 sections = []
    sections = []

    # 2. 循环每行的列表
    for line in lines:
        strip_line = line.strip()
        # 2.1 判断代码块状态
        if strip_line .startswith('```') or strip_line .startswith('~~~') :
            #第一次来一定进入代码块
            is_code_block = not is_code_block #取反即可
            #内容一定不是标题
            current_lines.append(line)
            continue
        # 2.2 判断是不是标题
        is_title = (not is_code_block) and re.match(title_pattern,strip_line) #是不是标题
        # 2.3 是标题怎么处理
        if is_title:
            # 先检查（是不是第一次），只要不是第一次就先存储
            if current_title :
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })
            current_title = strip_line
            current_lines = [current_title]
            title_count +=1
        #2.4 不是标题怎么处理
        else:
            current_lines.append(line)
    #最后一个标题内容保存
    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title
        })
    # 3. 返回结果sections
    logger.info(f"已经完成了chunks的语义粗切！识别chunk数量：{title_count},切片内容{sections}")
    return sections,title_count,len(lines)


def split_long_section(section, max_length):
    # 将当前chunk内容超长进行二次切割
    # 返回切割后的【{}，{}】
    # 1. content获取
    content = section.get("content")

    # 2. 判断content是否超长 没有的话直接返回
    if len(content) < max_length:
        logger.info(f"[split_long_section]:{content}当前chunk长度小于等于{max_length}，不做二次切割")
        return [section]

    # 3. 超长了，进行二次切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, #切割每块的最大长度
        chunk_overlap=100, #下次的重叠长度
        separators=['\n\n','\n','。','！',':'," "], #切割的符号（什么节点切割）
    )

    sub_sections = []
    for index,chunk in enumerate(splitter.split_text(content),start=1) :
        text = chunk.strip() #切片的内容
        title = f"{section.get('title')}_{index}"
        parent_title = section.get("parent_title")
        part = index
        file_title = section.get("file_name")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": parent_title,
            "part": part,
        })
    # 4. 返回切割后的结果
    return sub_sections


def merge_short_sections(final_sections, min_length):
    """
    上次切的太碎！ 还需要做合并
     1. content长度小于min_length
     2. 同一个parent_title才能合并
    :param final_sections:
    :param min_length:
    :return:
    """
    merged_sections = []
    pre_section =  None
    for section in final_sections:
        if pre_section is None:
            pre_section = section #第一次要处理的切片赋值给他！
            continue
        #pre_section 上次的块 ，section 本次的块
        is_pre_short = len (pre_section.get("content")) < min_length
        is_same_parent_title = pre_section.get("parent_title") and (pre_section.get("parent_title")) == section.get("parent_title")

        if is_pre_short and is_same_parent_title:
            # 上次既是短块，并且和本次是同一个父标题
            # 上次是短块 -》 和本次进行合并
            parent_section = section.get("parent_title")
            #pre_content = section.get("content")
            pre_section["content"] += "\n\n"+section.get("content")
            pre_section['part'] = section.get("part")
        else:
            #上次不是短块，或者和本次不是同一父标题
            merged_sections.append(section)
            pre_section = section  #本次变成上次
    if pre_section is not None:
        merged_sections.append(pre_section)

    return merged_sections

def step_3_refine_chunks(sections, max_length,min_length):
    """
    做内容的精细切割！
        1. 超过max_length块，要切割！ （parent_title|part）
        2. 小于min_length块，要合并结果！ （同一个parent_title）
    :param sections:
    :param max_length:
    :param min_length:
    :return:sections
    """
    final_sections = []
    # 超过的先切碎
    for section in sections:
        sub_sections = split_long_section(section,max_length)
        final_sections.extend(sub_sections)
    # 小于的再合并
    final_sections = merge_short_sections(final_sections,min_length)

    # 补全属性和参数
    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title')
    # 返回即可
    return final_sections


def step_4_backup_cunks(state, sections):
    """
    讲切割完的碎片进行存储！！
    :param state: 本地地址 local_dir
    :param sections: 要存的内容【{}】
    :return:
    """
    local_dir = state["local_dir"]
    backup_file_path = os.path.join(local_dir, "chunks.json")
    with open(backup_file_path,"w",encoding="utf-8") as f:
        json.dump(
            sections, #将什么数据写指定的文件流
            f, #写出的位置
            ensure_ascii=False, #中文直接原文存储
            indent=4 # json带有缩进4
        )
    logger.info(f"已经将内容进行备份，备份到：{backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为{state}")
    add_running_task(state["task_id"], function_name)

    try:
        #1. 参数校验
        md_content,file_title = step_1_get_content(state)

        #2. 粗粒度切割（md）语义完善 -》 使用标题切割
        #[{content:标题内容,title：标题,file_ title：文件名},{},{}]
        sections,title_count,lines_count = step2_split_by_title(md_content,file_title)

        #3. 特殊场景，一个文档没有标题，我们给他一个默认标题
        if title_count == 0:
            #证明没有标题
            sections = [{"title":"没有主题","content":md_content,"file_title":file_title}]

        #4. 细粒度切割（md）大小和重叠合适 -> 大 -》（设置重叠） 小 || 小 -》 合并 大小合适，语义完整的chunks
        sections = step_3_refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)

        #5. 数据的备份和chunks属性的修改
        state['chunks'] = sections
        step_4_backup_cunks(state,sections)
    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生了异常，异常信息为{e}")
        raise
    finally:
        logger.info(f">>> [{function_name}]结束执行了！现在的状态为{state}")
        add_done_task(state["task_id"], function_name)

    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")