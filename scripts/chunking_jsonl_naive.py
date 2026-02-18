from __future__ import annotations
import json
import hashlib
import pathlib
import re
from typing import List, Dict, Any

import tiktoken
from bs4 import BeautifulSoup

CONFIG = {
    "input_file": "/app/original_text/embedding_samples.jsonl",
    "output_file": "/app/data/chunks_all.jsonl",
    "max_chunk_tokens": 512,
    "chunk_overlap": 50,
    "min_merge_tokens": 50,     # 小于这个长度的段落会尝试向上或向下合并
    "min_tail_tokens": 30,      # chunk切分时，丢弃或合并过小的尾巴
    "min_words_per_block": 8,   
    "inject_context": True,     # 【新功能】开关：是否将路径写入 Embedding 文本
}

_enc = tiktoken.get_encoding("cl100k_base")

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def clean_wikipedia_garbage(text: str) -> str | None:
    if not text:
        return None

    garbage_triggers = [
        r"This article has multiple issues",
        r"Please help improve (it|this article)",
        r"discuss these issues on the talk page",
        r"Jump to\s*:\s*navigation",
        r"Edit links",
        r"About Wikipedia",
        r"This page was last edited",
        r"Hidden categories",
        r"Articles needing additional references",
        r"Statements consisting only of original research",
        r"verifying the claims made and adding inline citations",
        r"lead section does not adequately summarize",
        r"article possibly contains original research",
    ]
    trigger_regex = re.compile("|".join(garbage_triggers), re.IGNORECASE)
    if trigger_regex.search(text):
        return None

    text = re.sub(r"\(\s*hide\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*Learn how and when.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if "中文" in text and "English" in text and len(text) < 100:
        return None
    if len(text) < 5:
        return None

    return text

def is_semantic_block(text: str) -> bool:
    """
    Filter out TOC / nav / listy fragments.
    """
    words = text.split()
    if len(words) < CONFIG["min_words_per_block"]:
        return False

    # 【改进点 1】: 过滤纯目录结构 (例如 "2.1 History 2.2 Origins")
    # 如果包含大量 "数字.数字" 模式，视为垃圾 TOC
    digit_dot_digit_count = len(re.findall(r"\b\d+\.\d+\b", text))
    if digit_dot_digit_count >= 3 and len(words) < 50:
        return False

    # too non-linguistic (tables/menus)
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(len(text), 1) < 0.55:
        return False

    # lots of very short tokens (menus)
    short = sum(1 for w in words if len(w) <= 2)
    if short / max(len(words), 1) > 0.35:
        return False

    return True

def chunk_text(text: str, size: int, overlap: int, min_tail_tokens: int) -> List[str]:
    tokens = _enc.encode(text or "")
    if not tokens:
        return []
    if len(tokens) <= size:
        return [_enc.decode(tokens)]

    step = max(1, size - overlap)
    windows: List[List[int]] = []
    idx = 0
    while idx < len(tokens):
        window = tokens[idx : idx + size]
        if not window:
            break
        windows.append(window)
        idx += step

    # merge tiny tail chunk into previous for better embedding quality
    if len(windows) >= 2 and len(windows[-1]) < min_tail_tokens:
        windows[-2].extend(windows[-1])
        windows.pop()

    return [_enc.decode(w) for w in windows]

def parse_html_with_bs4(html_text: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_text, "html.parser")

    kill_selectors = [
        "#toc", ".toc",
        "#mw-navigation", "#footer",
        ".mw-editsection", "sup.reference",
        ".navbox", ".vertical-navbox", ".infobox", ".metadata", ".mbox", ".sidebar",
        "div.reflist", "ol.references",
        "table.infobox", "table.metadata",
    ]
    for sel in kill_selectors:
        for node in soup.select(sel):
            node.decompose()

    headers: Dict[int, str] = {}
    raw_sections: List[Dict[str, str]] = []

    tags_of_interest = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
    skip_header_patterns = [
        r"^references\b", r"^external links\b", r"^see also\b",
        r"^notes\b", r"^further reading\b", r"^bibliography\b",
    ]
    skip_header_re = re.compile("|".join(skip_header_patterns), re.IGNORECASE)

    for element in soup.find_all(tags_of_interest):
        tag_name = element.name.lower()

        if tag_name.startswith("h"):
            header_text = element.get_text(separator=" ", strip=True)
            header_text = re.sub(r"\[\d+\]", "", header_text).strip()
            header_text = re.sub(r"\(\s*edit\s*\)$", "", header_text, flags=re.IGNORECASE).strip()

            if header_text and skip_header_re.search(header_text):
                level = int(tag_name[1])
                headers[level] = "__SKIP__"
                for l in range(level + 1, 7):
                    headers.pop(l, None)
                continue

            if header_text:
                level = int(tag_name[1])
                headers[level] = header_text
                for l in range(level + 1, 7):
                    headers.pop(l, None)
            continue

        if any(h == "__SKIP__" for h in headers.values()):
            continue

        content_text = element.get_text(separator=" ", strip=True)
        cleaned_text = clean_wikipedia_garbage(content_text)
        if not cleaned_text:
            continue
        if not is_semantic_block(cleaned_text):
            continue

        active_headers = [headers[k] for k in sorted(headers.keys()) if headers[k] != "__SKIP__"]
        context_prefix = " > ".join(active_headers) if active_headers else "Summary"

        raw_sections.append({"context": context_prefix, "text": cleaned_text})

    return raw_sections

def merge_small_sections(
    sections: List[Dict[str, str]], min_tokens: int, max_tokens: int
) -> List[Dict[str, str]]:
    """
    【改进点 2】: 增强合并逻辑。
    1. 同 Context 合并 (原有逻辑)
    2. 如果当前段落是“过渡句”(以冒号结尾) 或 极短 (< 30 tokens)，强制与下一段合并，
       并采用下一段的 Context (防止 Context 漂移导致的孤儿节点)。
    """
    if not sections:
        return []
    
    merged: List[Dict[str, str]] = []
    current = sections[0]
    current_tokens = len(_enc.encode(current["text"]))

    for nxt in sections[1:]:
        nxt_tokens = len(_enc.encode(nxt["text"]))
        same_context = (current["context"] == nxt["context"])
        
        # 判定是否为“过渡性文本” (e.g., "The types are:", "Advantages include:")
        is_transition = current["text"].strip().endswith(":") and current_tokens < 30
        is_tiny = current_tokens < 30

        # 合并条件：
        # A. 同一 Context 下，如果不超长，就合并
        # B. 如果当前块是过渡句或极小，即使 Context 不同，也强行合并到下一块（作为下一块的前缀）
        should_merge = False
        
        if same_context:
            if current_tokens < min_tokens or (current_tokens + nxt_tokens <= max_tokens):
                should_merge = True
        elif is_transition or is_tiny:
            # 跨 Context 合并：如果当前块太小，为了不浪费，把它贴到下一块头上
            if current_tokens + nxt_tokens <= max_tokens:
                should_merge = True
                # 【关键】：当跨 Context 合并时，我们通常认为下一块的 Context 更具体或更重要
                # 所以我们把当前文本加到下一块，但保留下一块的 Context
                # 这里不需要显式改 current['context']，因为下面 current = nxt 会重置

        if should_merge:
            current["text"] += "\n" + nxt["text"]
            current_tokens += nxt_tokens
            # 如果是跨 Context 合并，通常希望保留更长/更新的那个 Context
            # 简单的做法是：如果合并了，Context 保持 `current` 的
            # 但如果是 is_transition 导致的合并，其实应该用 `nxt` 的 Context
            if not same_context:
                current["context"] = nxt["context"] 
        else:
            merged.append(current)
            current = nxt
            current_tokens = nxt_tokens

    merged.append(current)
    return merged

def process_record(obj: Dict[str, Any], order_base: int) -> List[Dict[str, Any]]:
    if obj.get("example_id") is not None:
        doc_id = str(obj["example_id"])
    elif obj.get("source"):
        doc_id = md5(str(obj["source"]))
    else:
        doc_id = str(order_base)

    text = obj.get("text") or ""
    results: List[Dict[str, Any]] = []

    raw_sections = parse_html_with_bs4(text)
    if not raw_sections:
        return []

    merged_sections = merge_small_sections(
        raw_sections,
        min_tokens=CONFIG["min_merge_tokens"],
        max_tokens=CONFIG["max_chunk_tokens"],
    )

    offset = 0
    for section in merged_sections:
        context_path = section["context"]
        base_text = section["text"]

        chunks = chunk_text(
            base_text,
            CONFIG["max_chunk_tokens"],
            CONFIG["chunk_overlap"],
            min_tail_tokens=CONFIG["min_tail_tokens"],
        )

        for chunk in chunks:
            token_count = len(_enc.encode(chunk))
            
            # 【改进点 3】: Context Injection (上下文注入)
            # 构造一个专门用于 Embedding 的文本，包含路径信息
            if CONFIG["inject_context"]:
                # 使用明确的分隔符，许多模型对此有优化
                text_for_embedding = f"Context: {context_path}\nContent: {chunk}"
            else:
                text_for_embedding = chunk

            md = {
                "doc_id": doc_id,
                "chunk_id": offset,
                "section_path": context_path,
                "tokens": token_count,
                "parser": "bs4_structure_chunker_v5_context_injected", # 更新版本号
            }
            
            # 保存时，text 字段是 Embedding 用的。
            # 如果你希望保留纯净原文用于展示，可以加一个 original_text 字段
            results.append({
                "text": text_for_embedding, 
                "original_text": chunk, # 给 LLM 阅读的纯净文本
                "metadata": md
            })
            offset += 1

    return results

def main():
    src = pathlib.Path(CONFIG["input_file"])
    dst = pathlib.Path(CONFIG["output_file"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"Start processing: {src} -> {dst}")

    try:
        with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
            for idx, line in enumerate(f_in, 1):
                if not line.strip():
                    continue
                try:
                    chunks = process_record(json.loads(line), idx * 100000)
                    for ch in chunks:
                        f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    total += len(chunks)
                    if idx % 100 == 0:
                        print(f"Processed {idx} lines...")
                except Exception as e:
                    print(f"Skipping line {idx}: {e}")
    except Exception as e:
        print(f"Fatal error: {e}")

    print(f"Done. Total chunks: {total}")

if __name__ == "__main__":
    main()