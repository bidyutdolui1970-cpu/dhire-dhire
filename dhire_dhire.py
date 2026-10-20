#!/usr/bin/env python3
"""Dhire Dhire: offline, resumable study-note Q&A generator.

This script reads a text file of notes and transforms it into a small set of
exam-style Q&A entries. It is intentionally lightweight: it works in a
checkpointed loop so the process can be resumed after interruptions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_MODEL = "gemma3:1b"
MAX_CHUNK_CHARS = 1800


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_notes(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = re.split(r"\n\s*\n+", cleaned)
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        para = paragraph.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= limit:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks or [cleaned]


def build_prompt(chunk: str) -> str:
    return (
        "You are generating exam study material from a student note. "
        "Return valid JSON only with an array named 'qa' where each item has: "
        "'question' and 'answer'. Keep questions clear and exam-style. Keep answers "
        "concise but complete. Add no extra commentary.\n\nNotes:\n"
        f"{chunk[:5000]}"
    )


def run_ollama(prompt: str, model: str) -> str:
    if shutil.which("ollama") is None:
        raise FileNotFoundError("ollama CLI not installed")

    result = subprocess.run(
        [
            "ollama",
            "run",
            model,
            prompt,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama failed")

    return result.stdout.strip()


def fallback_qa(chunk: str) -> List[Dict[str, str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    if not sentences:
        return []

    focused = sentences[:4]
    qa: List[Dict[str, str]] = []
    for idx, sentence in enumerate(focused, start=1):
        simplified = sentence.strip().rstrip(".")
        if len(simplified) < 25:
            continue
        question = f"What is the key point in the following note? ({idx})"
        answer = simplified
        qa.append({"question": question, "answer": answer})

    if not qa:
        qa.append({
            "question": "What is the central idea of the notes?",
            "answer": chunk[:400],
        })

    return qa


def parse_json_text(raw: str) -> List[Dict[str, str]]:
    text = raw.strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
        else:
            return []

    if isinstance(payload, dict):
        if isinstance(payload.get("qa"), list):
            return [item for item in payload["qa"] if isinstance(item, dict)]
        if isinstance(payload.get("questions"), list):
            return [item for item in payload["questions"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def generate_chunk_qas(chunk: str, model: str) -> List[Dict[str, str]]:
    try:
        output = run_ollama(build_prompt(chunk), model=model)
        qas = parse_json_text(output)
        if qas:
            return qas
    except Exception:
        pass

    return fallback_qa(chunk)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"completed_chunks": [], "qa": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"completed_chunks": [], "qa": []}
    return {
        "completed_chunks": data.get("completed_chunks", []),
        "qa": data.get("qa", []),
    }


def save_checkpoint(path: Path, completed_chunks: List[int], qa: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed_chunks": completed_chunks, "qa": qa}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_output_path(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Dhire Dhire offline study Q&A generator")
    parser.add_argument("--notes", required=True, help="Path to the notes file")
    parser.add_argument("--output", default="output/qa.json", help="Where to save the generated Q&A JSON")
    parser.add_argument("--checkpoint", default="output/checkpoint.json", help="Checkpoint file for resume support")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model name to use")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint instead of starting at the beginning")
    args = parser.parse_args()

    notes_path = Path(args.notes)
    output_path = build_output_path(args.output)
    checkpoint_path = Path(args.checkpoint)

    if not notes_path.exists():
        raise FileNotFoundError(f"Notes file not found: {notes_path}")

    notes = read_text(notes_path)
    chunks = split_notes(notes)
    checkpoint = load_checkpoint(checkpoint_path)

    completed = set(checkpoint.get("completed_chunks", []))
    all_qa: List[Dict[str, str]] = checkpoint.get("qa", [])

    if not args.resume:
        completed = set()
        all_qa = []
        save_checkpoint(checkpoint_path, [], [])

    for index, chunk in enumerate(chunks):
        if index in completed:
            continue

        generated = generate_chunk_qas(chunk, args.model)
        all_qa.extend(generated)
        completed.add(index)
        save_checkpoint(checkpoint_path, sorted(completed), all_qa)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"qa": all_qa}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated {len(all_qa)} Q&A entries from {len(chunks)} chunks.")
    print(f"Saved output to {output_path}")
    print(f"Checkpoint saved to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
