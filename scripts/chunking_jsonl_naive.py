"""
Industrial-Grade Structure-Aware Chunker.
Uses BeautifulSoup4 to robustly parse messy HTML, extract headers (H1-H6),
and inject hierarchical context into paragraphs and lists.
Fast, 0 cost, and highly optimized for Vector DB Retrieval.
"""
from __future__ import annotations
import json
import hashlib
import pathlib
from typing import List, Dict, Any

import tiktoken
from bs4 import BeautifulSoup

CONFIG = {
    "input_file": "/app/original_text/embedding_samples.jsonl",
    # Keep downstream ingest path stable
    "output_file": "/app/data/chunks_all.jsonl",
    "max_chunk_tokens": 512,
    "chunk_overlap": int(512 * 0.3),  # 30% overlap
}

_enc = tiktoken.get_encoding("cl100k_base")

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Token级的安全网：万一某个段落特别长，依然要保证它不超过 max_tokens"""
    tokens = _enc.encode(text or "")
    if not tokens:
        return []
    step = max(1, size - overlap)
    chunks: List[str] = []
    idx = 0
    while idx < len(tokens):
        window = tokens[idx : idx + size]
        if not window:
            break
        chunks.append(_enc.decode(window))
        idx += step
    return chunks

def parse_html_with_bs4(html_text: str) -> List[Dict[str, str]]:
    """
    使用 BeautifulSoup 解析 HTML。
    按照文档流的顺序遍历标签，维护当前的标题层级状态。
    返回结构字典：[{"context": "标题1 > 标题2", "text": "实际段落内容"}, ...]
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    
    headers = {}
    enriched_sections = []
    
    # 我们只关心构成文档结构的“骨架”标签和“血肉”标签
    tags_of_interest = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']
    
    # 按照在 HTML 文档中出现的顺序遍历这些标签
    for element in soup.find_all(tags_of_interest):
        tag_name = element.name
        
        if tag_name.startswith('h'):
            # 遇到标题：更新层级树
            level = int(tag_name[1])
            # 提取纯文本，去掉标题内可能包含的 <a> 或 <span>
            header_text = element.get_text(separator=" ", strip=True) 
            
            if header_text:
                headers[level] = header_text
                # 清除当前层级之下的所有子标题（例如遇到了新的 H2，就清空 H3, H4）
                for l in range(level + 1, 7):
                    headers.pop(l, None)
                    
        else:
            # 遇到段落或列表：提取内容
            # separator=" " 保证列表项 <li> 之间有空格，不会粘连
            content_text = element.get_text(separator=" ", strip=True)
            if not content_text:
                continue
                
            # 组装当前的上下文路径
            active_headers = [headers[l] for l in sorted(headers.keys())]
            context_prefix = " > ".join(active_headers)
            
            enriched_sections.append({
                "context": context_prefix,
                "text": content_text
            })
            
    return enriched_sections

def process_record(obj: Dict[str, Any], order_base: int) -> List[Dict[str, Any]]:
    doc_id = str(obj.get("example_id") or obj.get("source") or obj.get("query") or order_base)
    source = obj.get("source") or doc_id
    text = obj.get("text") or obj.get("document_text") or ""

    results: List[Dict[str, Any]] = []
    
    # 1. 结构化解析：提取自带标题上下文的段落
    sections = parse_html_with_bs4(text)
    
    # 如果纯文本没有命中任何 P 标签（可能是极其不规则的纯文本），做个兜底
    if not sections:
        clean_text = BeautifulSoup(text, 'html.parser').get_text(separator=" ", strip=True)
        sections = [{"context": "General", "text": clean_text}]
        
    # 2. 对每个带有结构上下文的段落进行 Token 级切分
    offset = 0
    for section in sections:
        context_path = section["context"]
        para_text = section["text"]
        
        # 将上下文拼接到文本前，提供给 Embedding 模型计算向量
        if context_path:
            full_text = f"[{context_path}]\n{para_text}"
        else:
            full_text = para_text
            
        chunks = chunk_text(full_text, CONFIG["max_chunk_tokens"], CONFIG["chunk_overlap"])
        
        for chunk in chunks:
            token_count = len(_enc.encode(chunk))
            
            # 【关键优化】：将 context_path 存入 metadata
            md = {
                "doc_id": doc_id,
                "source": source,
                "type": "text",
                "section_path": context_path,  # <- 方便向量数据库做 Metadata 过滤
                "page": 1,
                "pages": [1],
                "order": order_base + offset,
                "tokens": token_count,
                "hash": md5(f"{doc_id}|{order_base + offset}|{chunk[:50]}"),
                "parser": "jsonl_bs4_structural",
            }
            results.append({"text": chunk, "metadata": md})
            offset += 1
            
    return results

def main():
    src = pathlib.Path(CONFIG["input_file"])
    dst = pathlib.Path(CONFIG["output_file"])
    dst.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in, 1):
            obj = json.loads(line)
            chunks = process_record(obj, order_base=idx * 100000)
            for ch in chunks:
                f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            if idx % 10 == 0:
                print(f"[{idx}] 条记录已处理, 目前共计 chunks: {total_chunks}")

    print(f"完成。共计记录: {idx}, 总 chunks: {total_chunks}")
    print(f"输出至 -> {dst}")

if __name__ == "__main__":
    main()
