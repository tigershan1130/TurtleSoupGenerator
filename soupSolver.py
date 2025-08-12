import asyncio
import json
import configparser
import threading
import argparse
from typing import List
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from deepseek_llm import DeepSeekLLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.schema.messages import HumanMessage
import uuid
import sys

# ---------------- CLI 参数 ---------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", action="store_true", help="是否覆盖当前数据库")
    return parser.parse_args()

args = parse_args()

# ---------------- Load Puzzles ---------------- #
with open("puzzle_data.json", "r", encoding="utf-8") as f:
    puzzles = json.load(f)

# ---------------- Config and Setup ---------------- #
class SentenceTransformerEmbeddingWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

embedding_function = SentenceTransformerEmbeddingWrapper('BAAI/bge-large-zh-v1.5')

def get_embedding(text):
    return np.array(embedding_function.embed_query(text))

config = configparser.ConfigParser()
config.read("config.ini")
deepseek_api_key = config["DeepSeek"]["api_key"]
deepseek_api_base = config["DeepSeek"].get("api_base", "https://api.deepseek.com/v1")
deepseek_model_name = config["DeepSeek"].get("model", "deepseek-reasoner")

llm = DeepSeekLLM(api_key=deepseek_api_key, api_base=deepseek_api_base, model_name=deepseek_model_name, temperature=1.0, max_tokens=2000)

vector_db = Chroma(
    collection_name="puzzle_solver_result",
    embedding_function=embedding_function,
    persist_directory="puzzle_solver_db"
)

if args.override:
    print("[INFO] 清空数据库...")
    existing = vector_db._collection.get()
    if existing and "ids" in existing:
        vector_db._collection.delete(ids=existing["ids"])

# ---------------- Stop Flag ---------------- #
stop_flag = False

def listen_for_keypress():
    global stop_flag
    input("Press Enter at any time to stop evaluation...\n")
    stop_flag = True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def retry_ainvoke(messages):
    return await llm.ainvoke(messages)

def evaluate_similarity(guess, answer):
    guess_emb = get_embedding(guess)
    answer_emb = get_embedding(answer)
    sim = cosine_similarity([guess_emb], [answer_emb])[0][0]
    return float(sim)

def clean_text(text: str) -> str:
    return text.replace("。。", "。").strip()

async def safe_ainvoke(prompt):
    try:
        print("[Full Prompt]", prompt)
        messages = [HumanMessage(content=prompt)]
        return await retry_ainvoke(messages)
    except Exception as e:
        print(f"[FATAL ERROR] 多次重试失败: {e}")
        return type('MockResponse', (), {"content": "不知道"})()

async def get_yes_no_answer(puzzle, question):
    soup_face = clean_text(puzzle['soupFace'])
    prompt = f"""你是海龟汤推理助手。
题面如下：{soup_face}
请判断以下问题是否与最终谜底有关。
问题：\"{question}\"
只回答：是、不是 或 不知道。"""
    response = await safe_ainvoke(prompt)
    answer = response.content.strip().lower()
    if "不知道" in answer[:5]:
        return "不知道"
    return "是" if "是" in answer[:3] else "不是"

async def generate_question(puzzle, history):
    short_history = history[-10:]
    soup_face = clean_text(puzzle['soupFace'])
    if short_history:
        prompt = f"""你现在正在解一个推理谜题：{soup_face}
你之前提出的问题和答案包括：\"{"；".join([f'{q}=>{a}' for q, a, *_ in short_history])}\"
请基于这些信息，生成一个全新的、可以缩小谜底范围的是/不是类型的问题（不需要包含解释）。"""
    else:
        prompt = f"""你现在正在解一个推理谜题：{soup_face}
请直接生成一个有助于接近真相的是/不是类型问题，直接输出在双引号内，例如：\"死者是否提前报警？\""""
    response = await safe_ainvoke(prompt)
    return response.content.strip()

async def generate_guess(puzzle, history):
    short_history = history[-10:]
    soup_face = clean_text(puzzle['soupFace'])
    if short_history:
        qa_part = '基于以下提问和回答：' + '；'.join([f'{q}=>{a}' for q, a, *_ in short_history])
    else:
        qa_part = '你还未进行任何提问。'
    prompt = f"题面是：{soup_face}。\n{qa_part}\n请直接写出你认为的最终谜底，不需要解释过程。"
    response = await safe_ainvoke(prompt)
    return response.content.strip()

async def load_existing_history(puzzle_id: str):
    try:
        result = vector_db._collection.get(where={"id": puzzle_id})
        if result and "documents" in result and result["documents"]:
            doc = json.loads(result["documents"][0])
            return doc.get("history", []), doc.get("solved", False), doc.get("guesses_used", 0), len(doc.get("history", []))
    except Exception as e:
        print(f"[WARN] 无法加载历史记录: {e}")
    return [], False, 0, 0

async def test_one_puzzle(puzzle, max_guesses: int = 30):
    global stop_flag
    history, solved, guesses_used, questions_asked = await load_existing_history(puzzle["id"])
    if solved:
        print(f"[SKIP] 已解出：{puzzle['name']}")
        return None

    guess = ""
    similarity = 0.0

    for i in range(questions_asked, max_guesses):
        if stop_flag:
            print("[STOPPED] 用户终止了执行。")
            break
        question = await generate_question(puzzle, history)
        answer = await get_yes_no_answer(puzzle, question)
        print(f"[Q{i+1}] {question} => {answer}")
        guess = await generate_guess(puzzle, history + [(question, answer)])
        similarity = evaluate_similarity(guess, puzzle['soupBase'])
        print(f"[Guess {i+1}] similarity={similarity:.2f}\n{guess}\n")
        history.append((question, answer, guess, similarity))

        if similarity >= 0.8:
            result = {
                "id": puzzle['id'],
                "name": puzzle['name'],
                "solved": True,
                "guesses_used": i + 1,
                "final_guess": guess,
                "similarity": float(similarity),
                "history": history
            }
            vector_db.add_documents([Document(
                page_content=json.dumps(result, ensure_ascii=False),
                metadata={"id": result['id'], "name": result['name'], "solved": True, "questions_asked": len(history)}
            )])
            return result

    result = {
        "id": puzzle['id'],
        "name": puzzle['name'],
        "solved": False,
        "guesses_used": max_guesses,
        "final_guess": guess,
        "similarity": float(similarity),
        "history": history
    }
    vector_db.add_documents([Document(
        page_content=json.dumps(result, ensure_ascii=False),
        metadata={"id": result['id'], "name": result['name'], "solved": False, "questions_asked": len(history)}
    )])
    return result

async def main():
    global stop_flag
    stop_flag = False

    key_thread = threading.Thread(target=listen_for_keypress)
    key_thread.start()

    try:
        await llm.init_session()
        results = []
        for puzzle in puzzles:
            print(f"\n==== Solving: {puzzle['name']} ({puzzle['id']}) ====")
            result = await test_one_puzzle(puzzle, max_guesses=20)
            if result:
                results.append(result)
            if stop_flag:
                break
        with open("soup_guess_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n测试完成，结果保存至 soup_guess_results.json")
    finally:
        await llm.close()

if __name__ == "__main__":
    asyncio.run(main())
