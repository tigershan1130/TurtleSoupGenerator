import asyncio
import json
import csv
import uuid
import argparse
import configparser
import signal
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

# ---------------- Prompt 模板 ---------------- #
def build_puzzle_prompt(subject: str, variation_seed: int) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个擅长设计逻辑推理谜题（海龟汤类型）的助手。\"海龟汤\"是一种需要玩家通过不断提问来还原真相的解谜游戏。玩家只能得到\"是 / 不是 / 不知道\"这类简单回应。题目通常描述一个看似怪异或意外的场景，背后却隐藏着一个合乎逻辑的真相, 请保持所有推理在一个思维链。
注意：虽然这个游戏叫\"海龟汤\",但题目本身不需要与\"海龟\"有关。题材设定是悬疑恐怖题材。题目起码需要10个以上线索才能破解汤底。
好的海龟汤谜题通常具有以下几个特征:
1.信息不对称但合理:汤面提供的信息能激发推理兴趣，留下关键但模糊的线索;
2.反常设定但逻辑自治:谜底设定可能很反常(比如克隆人、心理暗示)，但应该有内部合理性;
3.顿悟时刻的“aha!”:揭示真相的一刻，玩家能感受到\"原来如此!\"而不是“啊?就这?
4.重读汤面后反而印证谜底:谜底不是生硬贴上的，而是让原本的疑问恰好被解释掉。
5.汤面必须有猜测主体，以及提问。
6.汤底必须是一句话概括整个汤面逻辑。
请生成与模板相关但不同设定的谜题变体，加入新的剧情和角色设定变化：{variation_seed}。
请确保所有 JSON 字符串中的引号都使用 \\\" 进行转义，避免语法错误。                     
请严格按照以下JSON格式返回:
```json
{{
    \"name\": \"题目简短名称 (10个字以内)\",
    \"soupFace\": \"题目表面（玩家看到的情况）\",
    \"soupBase\": \"真实真相（背后的推理真相）\",
    \"fewShots\": [
        \"问题1:回答1\",
        \"问题2:回答2\",
        \"问题3:回答3\",
        \"问题4:回答4\",
        \"问题5:回答5\"
    ],
    \"difficulty\": 1 (1: 简单， 2: 中等， 3: 很难推理)
}```"""),
        HumanMessage(content=f"""请根据以下模板生成一题，保留核心逻辑，但完全改变故事背景、角色设定与线索方式：{subject} 变化提示：{variation_seed}""")
    ])


def build_clue_prompt(soup_base: str, num_clues: int = 10) -> ChatPromptTemplate:
    """根据汤底生成线索的 Prompt。"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个逻辑校验助手，负责根据给定的汤底生成有助于玩家推理的线索。每条线索必须能够从汤底推导并附带对应的事实。"""),
        HumanMessage(content=f"""汤底如下：\n{soup_base}\n请提供{num_clues}条线索，并按以下JSON格式返回：\n```json\n{{\n  \"clues\": [\n    {{\"clue\": \"线索1\", \"fact\": \"事实1\"}},\n    {{\"clue\": \"线索2\", \"fact\": \"事实2\"}}\n  ]\n}}\n```""")
    ])


async def build_clue_from_fact(fact: str) -> str:
    """将汤底事实改写成简洁线索。"""
    messages = ChatPromptTemplate.from_messages([
        SystemMessage(content="""根据给定的汤底事实，改写成一句简洁的线索供玩家推理。可以省略直接指向真相的描述。"""),
        HumanMessage(content=f"""汤底事实：{fact}\n线索：""")
    ]).format_messages()
    try:
        result = await llm.ainvoke(messages)
        return result.content.strip()
    except Exception:
        return ""


async def verify_clue(clue: str, soup_base: str) -> bool:
    """验证单条线索是否能从汤底事实推出。"""
    messages = ChatPromptTemplate.from_messages([
        SystemMessage(content="""请判断给定的线索是否能从汤底推出。只回答 yes 或 no。"""),
        HumanMessage(content=f"""汤底：{soup_base}\n线索：{clue}\n这个线索能从汤底推出吗？""")
    ]).format_messages()
    try:
        result = await llm.ainvoke(messages)
        return "yes" in result.content.strip().lower()
    except Exception:
        return False

# ---------------- Prompt 文件读取 ---------------- #
def load_prompt_file(path: str, threshold: float = 12.5) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        riddles = json.load(f)
        return [r for r in riddles if r.get("score", 0) + r.get("humanScore", 0) >= threshold]

# ---------------- 数据保存 ---------------- #
def save_all_to_disk(json_file=OUTPUT_JSON, csv_file=OUTPUT_CSV):
    all_docs = vector_db._collection.get()
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

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, ensure_ascii=False, indent=2)
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
        for r in templates:
            source_id = r.get("id", str(uuid.uuid4())[:8])
            template = json.dumps({"name": r.get("name", ""), "soupFace": r.get("soupFace", ""), "soupBase": r.get("soupBase", "")}, ensure_ascii=False)
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

                    # 第二次调用模型生成线索
                    clue_messages = build_clue_prompt(data["soupBase"]).format_messages()
                    clue_result = await llm.ainvoke(clue_messages)
                    clue_content = clue_result.content.strip()
                    if clue_content.startswith("```json"):
                        clue_content = clue_content.strip("` \n")
                        clue_content = clue_content[clue_content.find("{") :]
                    clue_data = json.loads(clue_content)
                    
                    raw_facts = []
                    for c in clue_data.get("clues", []):
                        if isinstance(c, dict):
                            fact_text = c.get("fact") or c.get("clue") or ""
                        else:
                            fact_text = c
                        if fact_text:
                            raw_facts.append(fact_text)

                    generated_clues = []
                    for fact in raw_facts:
                        formatted = await build_clue_from_fact(fact)
                        if not formatted:
                            continue
                        if await verify_clue(formatted, data["soupBase"]):
                            generated_clues.append(formatted)

                    puzzle = {
                        "id": str(uuid.uuid4())[:8],
                        "name": data["name"],
                        "soupBase": data["soupBase"],
                        "soupFace": data["soupFace"],
                        "fewShots": data["fewShots"],
                        "clues": generated_clues,
                        "difficulty": data.get("difficulty", 2),
                        "sourceID": source_id
                    }
                    doc = Document(
                        page_content=json.dumps(puzzle, ensure_ascii=False),
                        metadata={"id": puzzle["id"], "source": "generated", "name": puzzle["name"], "sourceID": source_id}
                    )
                    vector_db.add_documents([doc])
                    print(f"[存储] 已保存 mutation {i+1} 的谜题：{puzzle['name']} (sourceID={source_id})")
                except Exception as e:
                    print("\n出[原始输]\n" + result.content + "\n")
                    print(f"[错误] 生成失败: {e}")
    finally:
        await llm.close()
        save_all_to_disk()

if __name__ == "__main__":
    asyncio.run(main())
