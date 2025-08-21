import asyncio
import json
import csv
import uuid
import argparse
import configparser
import signal
import re
from tqdm.asyncio import tqdm
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from deepseek_llm import DeepSeekLLM

# ---------------- 配置读取 ---------------- #
config = configparser.ConfigParser()
config.read("config.ini")
openAI_api_key = config["OpenAI"]["api_key"]
deepseek_api_key = config["DeepSeek"]["api_key"]
deepseek_api_base = config["DeepSeek"].get("api_base", "https://api.deepseek.com/v1")
deepseek_model_name = config["DeepSeek"].get("model", "deepseek-reasoner")

num_mutations = 5

# ---------------- 基本设定 ---------------- #
MODEL_NAME = "gpt-4-turbo-preview"
EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key=openAI_api_key)
OUTPUT_JSON = "puzzle_data.json"
OUTPUT_CSV = "puzzle_data.csv"
DB_PATH = "puzzle_db"

llm = DeepSeekLLM(api_key=deepseek_api_key, api_base=deepseek_api_base, model_name=deepseek_model_name, temperature=1.5, max_tokens=2000)

vector_db = Chroma(
    collection_name="puzzle_collection",
    embedding_function=EMBEDDING_MODEL,
    persist_directory=DB_PATH
)

# ---------------- 线索长度限制 ---------------- #
MAX_CLUE_LENGTH = 30  # 每条线索的最大字符数
MIN_CLUE_LENGTH = 5   # 每条线索的最小字符数

# ---------------- Prompt 模板 ---------------- #
def build_puzzle_prompt(subject: str, variation_seed: int) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个擅长设计逻辑推理谜题（海龟汤类型）的助手。\"海龟汤\"是一种需要玩家通过不断提问来还原真相的解谜游戏。玩家只能得到\"是 / 不是 / 不知道\"这类简单回应。题目通常描述一个看似怪异或意外的场景，背后却隐藏着一个合乎逻辑的真相, 请保持所有推理在一个思维链。
注意：虽然这个游戏叫\"海龟汤\",但题目本身不需要与\"海龟\"有关。题材设定是悬疑恐怖题材。题目起码需要10个以上线索才能破解汤底。
好的海龟汤谜题通常具有以下几个特征:
1.信息不对称但合理:汤面提供的信息能激发推理兴趣，留下关键但模糊的线索;
2.反常设定但逻辑自治:谜底设定可能很反常(比如克隆人、心理暗示)，但应该有内部合理性;
3.顿悟时刻的"aha!":揭示真相的一刻，玩家能感受到"原来如此!"而不是"啊?就这?
4.重读汤面后反而印证谜底:谜底不是生硬贴上的，而是让原本的疑问恰好被解释掉。
5.汤面必须有猜测主体，以及提问。
6.汤底必须是一句话概括整个汤面逻辑。
请生成与模板相关但不同设定的谜题变体，加入新的剧情和角色设定变化：{variation_seed}。
请确保所有 JSON 字符串中的引号都使用 \\" 进行转义，避免语法错误。                     
请严格按照以下JSON格式返回:
```json
{{
    "name": "题目简短名称 (10个字以内)",
    "soupFace": "题目表面（玩家看到的情况）",
    "soupBase": "真实真相（背后的推理真相）",
    "fewShots": [
        "问题1:回答1",
        "问题2:回答2",
        "problem3:answer3",
        "问题4:回答4",
        "问题5:回答5"
    ],
    "difficulty": 1 (1: 简单， 2: 中等， 3: 很难推理)
}```"""),
        HumanMessage(content=f"""请根据以下模板生成一题，保留核心逻辑，但完全改变故事背景、角色设定与线索方式：{subject} 变化提示：{variation_seed}""")
    ])

# ---------------- 线索提取函数 ---------------- #
def extract_detailed_clues(soup_base: str, min_clues: int = 15) -> List[str]:
    """从汤底文本中提取详细的短语作为线索"""
    # 清理汤底文本
    soup_base = re.sub(r'\s+', ' ', soup_base)
    
    # 第一步：按句子分割
    sentences = re.split(r'[。！？.!?]', soup_base)
    clues = [s.strip() for s in sentences if s.strip() and len(s.strip()) > MIN_CLUE_LENGTH]
    
    # 第二步：对每个句子进一步分割
    detailed_clues = []
    for clue in clues:
        # 按逗号、分号分割
        sub_clues = re.split(r'[，,；;]', clue)
        sub_clues = [s.strip() for s in sub_clues if s.strip() and len(s.strip()) > MIN_CLUE_LENGTH]
        
        # 检查长度，如果超过限制则进一步分割
        for sub_clue in sub_clues:
            if len(sub_clue) > MAX_CLUE_LENGTH:
                # 尝试按更细的标点分割
                smaller_clues = re.split(r'[:：、]', sub_clue)
                smaller_clues = [s.strip() for s in smaller_clues if s.strip() and len(s.strip()) > MIN_CLUE_LENGTH]
                detailed_clues.extend(smaller_clues)
            else:
                detailed_clues.append(sub_clue)
    
    # 第三步：提取括号内的内容作为独立线索
    parentheses_content = re.findall(r'[(（](.*?)[)）]', soup_base)
    parentheses_content = [s.strip() for s in parentheses_content if s.strip() and 
                          MIN_CLUE_LENGTH <= len(s.strip()) <= MAX_CLUE_LENGTH]
    detailed_clues.extend(parentheses_content)
    
    # 第四步：如果还不够，按其他标点符号分割
    if len(detailed_clues) < min_clues:
        more_clues = re.split(r'[:：、]', soup_base)
        more_clues = [s.strip() for s in more_clues if s.strip() and 
                     MIN_CLUE_LENGTH <= len(s.strip()) <= MAX_CLUE_LENGTH]
        detailed_clues.extend(more_clues)
    
    # 去重并确保线索长度在限制范围内
    unique_clues = []
    for clue in detailed_clues:
        if MIN_CLUE_LENGTH <= len(clue) <= MAX_CLUE_LENGTH:
            unique_clues.append(clue)
        elif len(clue) > MAX_CLUE_LENGTH:
            # 如果仍然太长，尝试更细的分割
            smaller_parts = re.split(r'[，,；;:：、]', clue)
            for part in smaller_parts:
                part = part.strip()
                if MIN_CLUE_LENGTH <= len(part) <= MAX_CLUE_LENGTH:
                    unique_clues.append(part)
    
    # 再次去重
    unique_clues = list(set(unique_clues))
    
    # 移除原始的长线索，只保留分割后的短线索
    final_clues = []
    for clue in unique_clues:
        # 检查这个线索是否是其他线索的一部分
        is_part_of_other = False
        for other_clue in unique_clues:
            if clue != other_clue and clue in other_clue:
                is_part_of_other = True
                break
        
        # 如果不是其他线索的一部分，或者它本身就是一个完整的线索，则保留
        if not is_part_of_other or len(clue) >= MIN_CLUE_LENGTH * 2:
            final_clues.append(clue)
    
    # 如果线索太多，截取前面的
    if len(final_clues) > min_clues * 2:
        final_clues = final_clues[:min_clues * 2]
    
    return final_clues

async def ensure_ten_clues(clues: List[str], soup_base: str) -> List[str]:
    """确保有10条线索，如果不足则使用LLM进一步分割"""
    if len(clues) >= 10:
        return clues
    
    # 构建提示，要求LLM从汤底中提取更多线索
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""你是一个海龟汤谜题设计助手。请从汤底中提取更多线索，确保总共有10条线索。
        
要求：
1. 线索必须是直接从汤底中提取的原文片段
2. 每条线索应该是独立的、简洁的事实片段
3. 每条线索长度应在{MIN_CLUE_LENGTH}-{MAX_CLUE_LENGTH}个字符之间
4. 不要创造任何新文字，只使用汤底中的原文

请确保提取的线索覆盖汤底的所有重要部分。"""),
        HumanMessage(content=f"""汤底：{soup_base}

当前已有{len(clues)}条线索：
{chr(10).join([f"{i+1}. {clue}" for i, clue in enumerate(clues)])}

请补充{10 - len(clues)}条线索，使总线索数达到10条。

请按照以下JSON格式返回：
{{"additional_clues": ["线索1", "线索2", ...]}}
""")
    ]).format_messages()
    
    try:
        result = await llm.ainvoke(prompt)
        content = result.content.strip()
        
        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        
        data = json.loads(content)
        additional_clues = data.get("additional_clues", [])
        
        # 合并线索并去重
        all_clues = list(set(clues + additional_clues))
        
        # 确保线索长度符合要求
        final_clues = [clue for clue in all_clues if MIN_CLUE_LENGTH <= len(clue) <= MAX_CLUE_LENGTH]
        
        return final_clues  # 确保不超过10条
    except Exception as e:
        print(f"补充线索时出错: {e}")
        return clues  # 如果出错，返回原始线索

async def select_best_clues(all_clues: List[str], soup_base: str, num_clues: int = 10) -> List[str]:
    """使用LLM从所有线索中筛选出最重要的num_clues条线索"""
    # 先过滤掉不符合长度要求的线索
    filtered_clues = [clue for clue in all_clues if MIN_CLUE_LENGTH <= len(clue) <= MAX_CLUE_LENGTH]
    
    if len(filtered_clues) <= num_clues:
        return filtered_clues
    
    # 构建提示
    clues_text = "\n".join([f"{i+1}. {clue}" for i, clue in enumerate(filtered_clues)])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""你是一个海龟汤谜题设计助手。请从提供的线索列表中选出{num_clues}条最重要的线索，这些线索应该能够帮助玩家逐步推理出真相。
        
选择标准：
1. 线索必须是直接从汤底中提取的原文片段
2. 每条线索应该是独立的、简洁的事实片段
3. 避免选择过于简短（少于{MIN_CLUE_LENGTH}字）或过于冗长（超过{MAX_CLUE_LENGTH}字）的线索
4. 优先选择包含关键信息的线索

请确保选择的线索符合长度要求，并且是直接从汤底中提取的原文。"""),
        HumanMessage(content=f"""汤底：{soup_base}

请从以下线索中选出{num_clues}条最重要的线索（保持原文不变）：
{clues_text}

请按照以下JSON格式返回：
{{"selected_clues": ["线索1", "线索2", ...]}}
""")
    ]).format_messages()
    
    try:
        result = await llm.ainvoke(prompt)
        content = result.content.strip()
        
        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        
        data = json.loads(content)
        selected_clues = data.get("selected_clues", [])
        
        # 确保选择的线索符合长度要求
        final_clues = [clue for clue in selected_clues if MIN_CLUE_LENGTH <= len(clue) <= MAX_CLUE_LENGTH]
        
        # 如果筛选后数量不足，补充一些原始线索
        if len(final_clues) < num_clues:
            additional_clues = [clue for clue in filtered_clues if clue not in final_clues]
            final_clues.extend(additional_clues[:num_clues - len(final_clues)])
        
        return final_clues
    except Exception as e:
        print(f"筛选线索时出错: {e}")
        return filtered_clues[:num_clues]

# ---------------- Prompt 文件读取 ---------------- #
def load_prompt_file(path: str, threshold: float = 12.5) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        riddles = json.load(f)
        return [r for r in riddles if r.get("score", 0) + r.get("humanScore", 0) >= threshold]

# ---------------- 数据保存 ---------------- #
def save_all_to_disk(json_file=OUTPUT_JSON, csv_file=OUTPUT_CSV):
    print(f"[DEBUG] 开始保存数据到磁盘...")
    
    try:
        all_docs = vector_db._collection.get()
        print(f"[DEBUG] 从向量数据库获取到 {len(all_docs.get('documents', []))} 个文档")
        
        puzzles = []
        for idx, doc in enumerate(all_docs.get("documents", [])):
            try:
                if not doc:
                    continue
                if all_docs["metadatas"][idx].get("metadata_field") == "token_usage":
                    continue
                parsed = json.loads(doc.strip())
                parsed["id"] = all_docs["metadatas"][idx].get("id", str(uuid.uuid4())[:8])
                parsed["sourceID"] = all_docs["metadatas"][idx].get("sourceID")
                puzzles.append(parsed)
            except Exception as e:
                print(f"[跳过] 第 {idx} 个文档解析失败: {e}")
                continue

        print(f"[DEBUG] 成功解析 {len(puzzles)} 个谜题")
        
        # 保存到JSON文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(puzzles, f, ensure_ascii=False, indent=2)
        
        # 保存到CSV文件
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "soupFace", "soupBase", "fewShots", "clues", "difficulty", "sourceID"])
            for p in puzzles:
                writer.writerow([
                    p["id"], p["name"], p["soupFace"], p["soupBase"],
                    "; ".join(p["fewShots"]), "; ".join(p["clues"]),
                    p.get("difficulty", 2),
                    p.get("sourceID", "")
                ])

        print(f"\n[终止保存] 已保存 {len(puzzles)} 条题目到 {json_file}, {csv_file}")
        
    except Exception as e:
        print(f"[错误] 保存到磁盘时出错: {e}")
        print(f"[错误] 错误类型: {type(e).__name__}")
        
        # 如果向量数据库有问题，尝试从本地文件读取
        try:
            print(f"[DEBUG] 尝试从本地文件 {json_file} 读取数据...")
            with open(json_file, "r", encoding="utf-8") as f:
                puzzles = json.load(f)
            print(f"[DEBUG] 从本地文件读取到 {len(puzzles)} 个谜题")
            print(f"\n[终止保存] 从本地文件读取到 {len(puzzles)} 条题目")
        except FileNotFoundError:
            print(f"[警告] 本地文件 {json_file} 不存在")
            print(f"\n[终止保存] 没有找到任何已保存的题目")
        except Exception as file_error:
            print(f"[错误] 读取本地文件时出错: {file_error}")
            print(f"\n[终止保存] 无法读取已保存的题目")

# ---------------- 调试工具 ---------------- #
def debug_print(message: str, debug_enabled: bool = False):
    """调试打印函数，只在启用调试时输出"""
    if debug_enabled:
        print(message)

# ---------------- 信号捕捉 ---------------- #
def setup_interrupt_handler():
    def handler(signum, frame):
        print("\n[中断] 收到终止信号，开始保存数据库内容...")
        save_all_to_disk()
        exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# ---------------- 主入口 ---------------- #
async def main():
    setup_interrupt_handler()
    await llm.init_session()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-json", type=str, default="scored_riddles.json")
    parser.add_argument("--mutations", type=int, default=num_mutations)
    parser.add_argument("--db-mode", choices=["append", "override"], default="append")
    parser.add_argument("--maxtemplates", type=int, help="最大处理的模板数量，用于提前终止循环")
    parser.add_argument("--debug", action="store_true", help="启用调试输出")
    args = parser.parse_args()

    if args.db_mode == "override":
        print("[重置] 清空现有数据库...")
        existing_docs = vector_db._collection.get()
        if existing_docs and existing_docs["ids"]:
            vector_db._collection.delete(ids=existing_docs["ids"])

    templates = load_prompt_file(args.prompt_json, threshold=12.5)
    existing_docs = vector_db._collection.get()
    existing_by_source = {}
    for meta in existing_docs.get("metadatas", []):
        sid = meta.get("sourceID")
        if sid:
            existing_by_source[sid] = existing_by_source.get(sid, 0) + 1

    print(f"当前过滤后模板数量{len(templates)}, mutations: {args.mutations}")

    try:
        processed_count = 0
        for r in templates:
            if args.maxtemplates and processed_count >= args.maxtemplates:
                print(f"[提前终止] 已达到最大模板处理数量: {args.maxtemplates}")
                break
            processed_count += 1
            source_id = r.get("id", str(uuid.uuid4())[:8])
            template = json.dumps({"name": r.get("name", ""), "soupFace": r.get("soupFace", ""), "soupBase": r.get("soupBase", "")}, ensure_ascii=False)
            debug_print(f"[DEBUG] 模板 {source_id} 原始汤底: {r.get('soupBase', '')}", args.debug)
            current_count = existing_by_source.get(source_id, 0)

            for i in range(current_count, args.mutations):
                print(f"[生成] 模板 {source_id} - mutation {i+1}/{args.mutations}")
                variation_seed = f"{source_id}-{i}"
                messages = build_puzzle_prompt(template, variation_seed).format_messages()
                try:
                    result = await llm.ainvoke(messages)
                    content = result.content.strip()
                    if content.startswith("```json"):
                        content = content.strip("` \n")
                        content = content[content.find("{") :]
                    data = json.loads(content)

                    debug_print(f"[DEBUG] 题目名称: {data['name']}", args.debug) 
                    debug_print(f"[DEBUG] 汤底 (soupBase): {data['soupBase']}", args.debug)
                    debug_print(f"[DEBUG] 汤面 (soupFace): {data['soupFace']}", args.debug)
                    
                    # 提取详细线索
                    debug_print(f"[DEBUG] 开始提取详细线索...", args.debug)
                    all_clues = extract_detailed_clues(data["soupBase"], min_clues=15)
                    debug_print(f"[DEBUG] 所有提取的线索: {all_clues}", args.debug)
                    
                    # 确保有10条线索
                    if len(all_clues) < 10:
                        debug_print(f"[DEBUG] 线索数量不足({len(all_clues)})，开始补充线索...", args.debug)
                        try:
                            all_clues = await ensure_ten_clues(all_clues, data["soupBase"])
                            debug_print(f"[DEBUG] 补充后的线索: {all_clues}", args.debug)
                        except Exception as clue_error:
                            debug_print(f"[DEBUG] 补充线索时出错: {clue_error}", args.debug)
                    
                    # 使用LLM筛选最重要的10条线索
                    debug_print(f"[DEBUG] 开始筛选最佳线索...", args.debug)
                    try:
                        selected_clues = await select_best_clues(all_clues, data["soupBase"])
                        debug_print(f"[DEBUG] 筛选后的线索: {selected_clues}", args.debug)
                    except Exception as select_error:
                        debug_print(f"[DEBUG] 筛选线索时出错: {select_error}", args.debug)
                        # 如果筛选失败，使用前10条线索
                        selected_clues = all_clues[:10]

                    puzzle = {
                        "id": str(uuid.uuid4())[:8],
                        "name": data["name"],
                        "soupBase": data["soupBase"],
                        "soupFace": data["soupFace"],
                        "fewShots": data["fewShots"],
                        "clues": selected_clues,
                        "difficulty": data.get("difficulty", 2),
                        "sourceID": source_id
                    }
                    
                    debug_print(f"[DEBUG] 最终保存的完整谜题: {json.dumps(puzzle, ensure_ascii=False, indent=2)}", args.debug)
                    
                    # 尝试保存到向量数据库
                    try:
                        doc = Document(
                            page_content=json.dumps(puzzle, ensure_ascii=False),
                            metadata={"id": puzzle["id"], "source": "generated", "name": puzzle["name"], "sourceID": source_id}
                        )
                        vector_db.add_documents([doc])
                        debug_print(f"[存储] 已保存 mutation {i+1} 的谜题到向量数据库：{puzzle['name']} (sourceID={source_id})", args.debug)
                    except Exception as db_error:
                        debug_print(f"[DEBUG] 向量数据库保存失败: {db_error}", args.debug)
                        # 即使数据库保存失败，也尝试保存到本地文件作为备份    
                        backup_file = "puzzle_backup.json"
                        # 读取现有备份
                        try:
                            with open(backup_file, "r", encoding="utf-8") as f:
                                backup_data = json.load(f)
                        except FileNotFoundError:
                            backup_data = []
                        
                        # 添加新数据
                        backup_data.append(puzzle)
                        
                        # 保存备份
                        with open(backup_file, "w", encoding="utf-8") as f:
                            json.dump(backup_data, f, ensure_ascii=False, indent=2)
                        
                        debug_print(f"[备份] 已将谜题保存到本地备份文件: {backup_file}", args.debug)
                        
                except Exception as e:
                    current_count = vector_db._collection.count()
                    debug_print(f"[DEBUG] 当前向量数据库中的文档数量: {current_count}", args.debug)
                    debug_print(f"[DEBUG] 捕获到异常: {type(e).__name__}", args.debug)
                    debug_print(f"[DEBUG] 异常详情: {str(e)}", args.debug)
                    if 'result' in locals() and hasattr(result, 'content'):
                        debug_print("\n[原始输出]\n" + result.content + "\n", args.debug)
                    debug_print(f"[错误] 生成失败: {e}", args.debug)

        debug_print(f"[DEBUG] 已处理 {processed_count} 个模板", args.debug)
    
    finally:
        await llm.close()
        save_all_to_disk()

if __name__ == "__main__":
    asyncio.run(main())