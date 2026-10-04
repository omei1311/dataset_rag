import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from langgraph.store.base import get_text_at_path
from openai import timeout
from pandas.core.groupby.ops import extract_result
from rich.json import JSON

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.path_util import PROJECT_ROOT
from app.conf.mineru_config import mineru_config


def step_1_validate_paths(state):
    logger.info(f">>> [step_1_validate_paths]在md转pdf下，开始进行文件格式校验！")

    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]
    if not pdf_path:
        logger.error("[step_1_validate_paths]检查发现没有输入文件，无法继续进行解析！")
        raise ValueError("[step_1_validate_paths]检查发现没有输入文件，无法继续进行解析！")
    if not local_dir:
        #给与一个输出的默认值
        local_dir = str(PROJECT_ROOT / "output")
        logger.info("[step_1_validate_paths]检查发现local_dir没有赋值，给与默认值！{local_dir}")
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths]检查发现pdf_path不存在，请检查输入文件路径是否正确！")
        raise FileExistsError(f"[step_1_validate_paths]检查发现pdf_path不存在，请检查输入文件路径是否正确！")
    if not local_dir_obj.exists():
        logger.error(f"[step_1_validate_paths]检查发现local_dir不存在，主动创建对应的文件夹！")
        local_dir_obj.mkdir(parents=True,exist_ok=True)

    return pdf_path_obj,local_dir_obj


def step_2_upload_and_poll(pdf_path_obj):
    #1. 申请上传解析的地址
    # 前置准备的参数 url api |token | 准备固定格式的请求头
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"

    logger.info(f"调试：请求URL = {url}")
    logger.info(f"调试：API Key = {token[:10]}...")  # 只打印前10位，避免泄露

    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [{"name": f"{pdf_path_obj.name}"}],
        "model_version": "vlm"
    }

    # 调试：打印请求数据
    logger.info(f"调试：请求数据 = {data}")

    response = requests.post(url, headers=header, json=data)

    # 调试：打印响应内容
    logger.info(f"调试：响应状态码 = {response.status_code}")
    logger.info(f"调试：响应内容 = {response.text}")

    if response.status_code != 200 or response.json()["code"] != 0:
        logger.error(f"请求失败，状态码：{response.status_code}，响应：{response.text}")
        raise RuntimeError(f"请求minerU解析接口失败，响应：{response.text}")
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}",}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url, headers=header, json=data)
    # 结果处理 请求http状态码不是200 或者返回结果的状态码不是0 请求失败！
    if response.status_code != 200 or response.json()["code"] != 0:
        logger.error(f"[step_2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确！")
        raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确！")
    upload_url = response.json()['data']['file_urls'][0]
    batch_id =  response.json()['data']['batch_id']  #处理id，后续根据这个id获取结果

    # 2.将文件上传到对应的解析地址
    #使用put请求，将pdf_path_obj文件传递到upload_url地址
    #不能直接使用put！
    http_session = requests.Session()
    http_session.trust_env = False
    try:
        with open(pdf_path_obj,'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(upload_url,data = file_data )
        if response.status_code != 200 or response.json()["code"] != 0:
            logger.error(f"[step_2_upload_and_poll]上传文件到MinerU失败，请检查输入文件路径是否正确！")
            raise RuntimeError(f"[step_2_upload_and_poll]上传文件到MinerU失败，请检查输入文件路径是否正确！")
    except Exception as e:
        logger.error(f"[step_2_upload_and_poll]上传文件到MinerU失败，请检查输入文件路径是否正确！")
        raise RuntimeError(f"[step_2_upload_and_poll]上传文件到MinerU失败，请检查输入文件路径是否正确！")
    finally:
        http_session.close()
    #3.轮询获取解析结果
    #循环获取！确保获取到结果，再执行
    #设计一个循环，3秒获取一次！最多等到10分钟600秒
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval = 3
    start_time = time.time()

    while True:
        #3.1 超时判断 不能站在第一次角度！站在宏观角度
        if time.time() - start_time > timeout_seconds:
            logger.error(f"[step_2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确！")
            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确！")

        #3.2 向指定的url地址获取本次解析的结果
        res = requests.get(url, headers=header)
        #3.3 解析结果判断和获取zip_url
        if res.status_code != 200 :
            if 500 <= res.status_code < 600 :
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口超时，返回状态码：{res.status_code}")
        json_data = res.json()
        if json_data['code'] != 0:
            #很大概率没有token了
            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口超时，返回错误：{json_data['code']}信息{json_data['msg']}")
        #判断解析状态
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            full_zip_url = extract_result['full_zip_url']
            logger.info(f"已经完成pdf的解析，耗时：{time.time()-start_time}s,解析结果：{full_zip_url}")
            return full_zip_url
        else:
            time.sleep(poll_interval)


def step_3_dowload_and_extract(zip_url,local_dir_obj,stem) ->str:

    # 1. 下载zip文件response响应体
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step_3_dowload_and_extract]下载文件失败，请检查输入文件路径是否正确！")
        raise RuntimeError(f"[step_3_dowload_and_extract]下载文件失败，请检查输入文件路径是否正确！")

    # 2. 响应体的zip文件保存到本地
    zip_sava_path = local_dir_obj /f"{stem}_result.zip"
    with open(zip_sava_path,'wb') as f:
        f.write(response.content)
    logger.info(f"[step_3_dowload_and_extract]下载文件成功，保存位置：{zip_sava_path}")

    # 3. 清空旧目录（将上次处理的文件目录进行删除）
    extract_target_dir = local_dir_obj /stem

    if extract_target_dir.exists():
        #递归删除目录内容
        shutil.rmtree(extract_target_dir)
    #创建一个新的目录
    extract_target_dir.mkdir(parents=True,exist_ok=True)

    # 4. 进行zip文件的解压工作
    #创建一个zip文件对象，只能读取，解压
    with zipfile.ZipFile(zip_sava_path,'r') as zip_file_object:
        #调用对象的解压方法进行解压 参数：解压的目标文件夹 output/二狗子
        zip_file_object.extractall(extract_target_dir)

    # 5. 返回md文件地址
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        logger.error(f"[step_3_dowload_and_extract]没有找到md文件，请检查输入文件路径是否正确！")
        raise RuntimeError(f"[step_3_dowload_and_extract]没有找到md文件，请检查输入文件路径是否正确！")

    target_md_file =  None #存储最终md文件
    for md_file in md_file_list:
        if target_md_file == stem + ".md":
            target_md_file = md_file
            break

    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break

    if not target_md_file:
        target_md_file = md_file_list[0]

    if target_md_file != stem + ".md":
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    #最终的md文件获取绝对路径，并返回字符串类型
    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"[step_3_dowload_and_extract]完成md解压，最终存储md路径为：{final_md_str_path}")
    return final_md_str_path

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 1. 进入节点的日志输出【节点+核心参数】 记录任务状态【哪个任务开始了】-》给前端推送消息（埋点）
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为{state}")
    add_running_task(state["task_id"], function_name)

    try:
        #校验
        pdf_path_obj,local_dir_obj = step_1_validate_paths(state)

        #参数：要解析的pdf文件路径  返回值：要下载的zip文件地址
        zip_url = step_2_upload_and_poll(pdf_path_obj)

        #参数： 要下载的地址 2.local_dir_obj 解析的文件夹 3.文件名二狗子（二狗子.pdf）
        md_path = step_3_dowload_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)

        state["md_path"] = md_path
        state["local_dir"] = str(local_dir_obj)
        with open(md_path,'r',encoding='utf-8') as f:
            state["md_content"] = f.read()

    except Exception as e:
        #处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生了异常，异常信息为{e}")
        raise
    finally:
        logger.info(f">>> [{function_name}]结束执行了！现在的状态为{state}")
        add_done_task(state["task_id"], function_name)

    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")