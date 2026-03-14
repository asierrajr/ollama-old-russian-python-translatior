#!/usr/bin/env python3
"""
Translate Pavlov-era / pre-1918 Russian OCR text to a target language using Ollama.

Features:
- Optional normalization with Qwen before translation
- Skip normalization entirely with --skip-normalization
- Resume interrupted runs with --resume
- Target language flags for code and prompt wording
- Incremental writes to <filename>_<target>.txt after each completed chunk
- Progress state in <filename>_progress.json
- Detects summary / wrong-language outputs and retries automatically
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, List

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_NORMALIZER_MODEL = "qwen3:30b"
DEFAULT_TRANSLATOR_MODEL = "translategemma:12b"
DEFAULT_TARGET_LANG_CODE = "en"
DEFAULT_TARGET_LANG_NAME = "English"

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
SUMMARY_MARKERS = [
    "основные моменты",
    "общая информация",
    "основные результаты",
    "выводы",
    "этот текст",
    "в целом",
    "я проанализировал",
    "предоставленный текст",
]

CYRILLIC_TARGET_CODES = {
    "ru", "ru-ru", "uk", "uk-ua", "be", "be-by", "bg", "bg-bg",
    "sr", "sr-rs", "mk", "mk-mk", "kk", "ky", "mn", "tg"
}
CYRILLIC_TARGET_NAMES = {
    "russian", "ukrainian", "belarusian", "bulgarian", "serbian",
    "macedonian", "kazakh", "kyrgyz", "mongolian", "tajik"
}


def sanitize_lang_code(lang_code: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", lang_code.strip())
    return cleaned or "translated"


def target_language_uses_cyrillic(target_lang_code: str, target_lang_name: str) -> bool:
    code = target_lang_code.strip().lower()
    name = target_lang_name.strip().lower()
    return code in CYRILLIC_TARGET_CODES or name in CYRILLIC_TARGET_NAMES


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def eprint(msg: str) -> None:
    print(f"[{ts()}] {msg}", file=sys.stderr, flush=True)


def read_text_file(path: pathlib.Path) -> str:
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode {path} as UTF-8 or CP1251") from last_error


def write_text_file(path: pathlib.Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def collapse_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, chunk_chars: int) -> List[str]:
    text = collapse_blank_lines(text)
    if len(text) <= chunk_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= chunk_chars:
            addition = len(para) + (2 if current else 0)
            if current and current_len + addition > chunk_chars:
                flush_current()
            current.append(para)
            current_len += addition
            continue

        flush_current()

        pieces = re.split(r"(?<=[\.\!\?\n])\s+", para)
        buf = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if not buf:
                if len(piece) <= chunk_chars:
                    buf = piece
                else:
                    for i in range(0, len(piece), chunk_chars):
                        chunks.append(piece[i:i + chunk_chars].strip())
                    buf = ""
            elif len(buf) + 1 + len(piece) <= chunk_chars:
                buf += " " + piece
            else:
                chunks.append(buf.strip())
                if len(piece) <= chunk_chars:
                    buf = piece
                else:
                    for i in range(0, len(piece), chunk_chars):
                        chunks.append(piece[i:i + chunk_chars].strip())
                    buf = ""
        if buf:
            chunks.append(buf.strip())

    flush_current()
    return [c for c in chunks if c.strip()]


def split_for_salvage(text: str, max_chars: int = 1200) -> List[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            out.append("\n".join(cur).strip())
            cur = []
            cur_len = 0

    for line in lines:
        line_len = len(line) + (1 if cur else 0)
        if cur and cur_len + line_len > max_chars:
            flush()
        if len(line) > max_chars:
            flush()
            for i in range(0, len(line), max_chars):
                out.append(line[i:i + max_chars].strip())
            continue
        cur.append(line)
        cur_len += line_len
    flush()
    return [x for x in out if x.strip()]


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    *,
    temperature: float,
    keep_alive: str,
    timeout: int,
    think: bool = False,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "think": think,
        "options": {
            "temperature": temperature,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Is `ollama serve` running?"
        ) from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected Ollama response: {raw[:500]}") from exc

    response = result.get("response", "")
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"Model {model} returned an empty response: {result}")
    return response.strip()


def normalize_old_russian(
    text: str,
    host: str,
    model: str,
    temperature: float,
    keep_alive: str,
    timeout: int,
) -> str:
    prompt = f"""You are an expert in late-Imperial and early 20th-century Russian.
The text below is OCR from Pavlov-era Russian and may contain pre-1918 spelling, archaic vocabulary, and OCR mistakes.

Task:
1. Restore only obvious OCR errors conservatively.
2. Convert pre-1918 spelling to modern Russian spelling.
3. Preserve meaning, names, dates, numbers, and technical terms.
4. Do not summarize.
5. Output only the normalized modern Russian text.

TEXT:
{text}
"""
    return ollama_generate(
        host,
        model,
        prompt,
        temperature=temperature,
        keep_alive=keep_alive,
        timeout=timeout,
        think=False,
    )


def looks_like_bad_translation(
    text: str,
    target_lang_code: str = "en",
    target_lang_name: str = "English",
) -> bool:
    t = text.strip().lower()

    if not t:
        return True

    normalized_code = (target_lang_code or "").lower()
    normalized_name = (target_lang_name or "").lower()

    # Summary / analysis markers are always bad for this workflow.
    if any(marker in t for marker in SUMMARY_MARKERS):
        return True

    bullet_lines = sum(
        1 for line in text.splitlines()
        if line.strip().startswith(("*", "-", "•"))
    )
    if bullet_lines >= 3:
        return True

    # Only use the "too much Cyrillic" heuristic for non-Cyrillic targets.
    cyrillic_targets = {
        "ru", "ru-ru", "uk", "uk-ua", "bg", "bg-bg", "sr", "sr-rs", "mk", "mk-mk", "be", "be-by"
    }
    if normalized_code not in cyrillic_targets and "cyril" not in normalized_name:
        cyr = len(CYRILLIC_RE.findall(text))
        letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
        if letters and (cyr / letters) > 0.08:
            return True

    return False


def build_translation_prompt(
    text: str,
    target_lang_code: str = "en",
    target_lang_name: str = "English",
    retry: bool = False,
) -> str:
    if not retry:
        return f"""You are a professional Russian (ru) to {target_lang_name} ({target_lang_code}) translator.
Your goal is to accurately convey the meaning and nuances of the original Russian text while adhering to {target_lang_name} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_lang_name} translation, without any additional explanations or commentary.
Translate every line of the source. Do not summarize. Do not omit headings, numbers, dates, initials, or table-like lines.

Please translate the following Russian text into {target_lang_name}:

{text}
"""
    return f"""You are a literal translation engine.

Translate the Russian OCR text below into {target_lang_name}.

Rules:
- Translate ALL content.
- Do NOT summarize.
- Do NOT explain.
- Do NOT paraphrase.
- Do NOT answer in Russian.
- Do NOT add bullet points, commentary, or introductions.
- Preserve headings, paragraph breaks, labels, dates, names, initials, and numbers.
- Preserve table-like lines as plain text.
- Translate line-by-line whenever possible.
- Do NOT omit repeated or awkward lines just because they look like OCR noise.
- If a fragment is damaged by OCR, translate what is readable and keep unclear text in [unclear OCR].
- Output {target_lang_name} only.

SOURCE TEXT:
<<<
{text}
>>>
"""


def translate_once(
    text: str,
    host: str,
    model: str,
    temperature: float,
    keep_alive: str,
    timeout: int,
    target_lang_code: str,
    target_lang_name: str,
    retry_prompt: bool,
) -> str:
    prompt = build_translation_prompt(
        text,
        target_lang_code=target_lang_code,
        target_lang_name=target_lang_name,
        retry=retry_prompt,
    )
    temp = 0.0 if retry_prompt else temperature
    return ollama_generate(
        host,
        model,
        prompt,
        temperature=temp,
        keep_alive=keep_alive,
        timeout=timeout,
        think=False,
    )


def translate_russian_to_english(
    text: str,
    host: str,
    model: str,
    temperature: float,
    keep_alive: str,
    timeout: int,
    target_lang_code: str,
    target_lang_name: str,
) -> str:
    out = translate_once(
        text,
        host,
        model,
        temperature,
        keep_alive,
        timeout,
        target_lang_code,
        target_lang_name,
        retry_prompt=False,
    )
    if not looks_like_bad_translation(out, target_lang_code, target_lang_name):
        return out.strip()

    eprint("Detected summary / wrong-language output; retrying chunk with stricter prompt...")
    out = translate_once(
        text,
        host,
        model,
        0.0,
        keep_alive,
        timeout,
        target_lang_code,
        target_lang_name,
        retry_prompt=True,
    )
    if not looks_like_bad_translation(out, target_lang_code, target_lang_name):
        return out.strip()

    eprint("Chunk still looked wrong; salvaging with smaller sub-chunks...")
    subparts = split_for_salvage(text, max_chars=1200)
    english_parts: List[str] = []
    for sub_idx, sub in enumerate(subparts, start=1):
        eprint(f"  Salvage sub-chunk {sub_idx}/{len(subparts)}...")
        sub_out = translate_once(
            sub,
            host,
            model,
            0.0,
            keep_alive,
            timeout,
            target_lang_code,
            target_lang_name,
            retry_prompt=True,
        )
        if looks_like_bad_translation(sub_out, target_lang_code, target_lang_name):
            raise RuntimeError(
                "Model returned a summary / wrong-language output instead of a translation."
            )
        english_parts.append(sub_out.strip())
    return "\n\n".join(english_parts).strip()


def build_output_paths(
    input_path: pathlib.Path,
    target_lang_code: str,
    output_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    stem = input_path.stem
    parent = output_dir if output_dir is not None else input_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    lang_suffix = sanitize_lang_code(target_lang_code)
    return (
        parent / f"{stem}_normalized_ru.txt",
        parent / f"{stem}_{lang_suffix}.txt",
        parent / f"{stem}_progress.json",
    )


def list_text_files(input_dir: pathlib.Path) -> List[pathlib.Path]:
    return [path for path in sorted(input_dir.glob("*.txt")) if path.is_file()]


def save_progress(
    progress_path: pathlib.Path,
    *,
    input_file: str,
    chunk_chars: int,
    total_chunks: int,
    skip_normalization: bool,
    target_lang_code: str,
    target_lang_name: str,
    completed_chunks: int,
    normalized_chunks: List[str],
    english_chunks: List[str],
) -> None:
    payload = {
        "input_file": input_file,
        "chunk_chars": chunk_chars,
        "total_chunks": total_chunks,
        "skip_normalization": skip_normalization,
        "target_lang_code": target_lang_code,
        "target_lang_name": target_lang_name,
        "completed_chunks": completed_chunks,
        "normalized_chunks": normalized_chunks,
        "english_chunks": english_chunks,
        "updated_at": ts(),
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_progress(progress_path: pathlib.Path) -> dict:
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read progress file: {progress_path}") from exc


def process_file(
    *,
    input_path: pathlib.Path,
    host: str,
    normalizer_model: str,
    translator_model: str,
    target_lang_code: str,
    target_lang_name: str,
    chunk_chars: int,
    temperature: float,
    keep_alive: str,
    timeout: int,
    save_normalized: bool,
    skip_normalization: bool,
    resume: bool,
    output_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    source_text = read_text_file(input_path)
    if not source_text.strip():
        raise RuntimeError(f"Input file is empty: {input_path}")

    normalized_path, english_path, progress_path = build_output_paths(input_path, target_lang_code, output_dir)
    chunks = split_into_chunks(source_text, chunk_chars)
    total = len(chunks)
    eprint(f"Loaded {input_path} ({len(source_text):,} chars) in {total} chunk(s).")

    normalized_chunks: List[str] = []
    english_chunks: List[str] = []
    start_idx = 0

    if progress_path.exists():
        if not resume:
            raise RuntimeError(
                f"Progress file already exists: {progress_path}. Use --resume to continue or delete it to start over."
            )
        state = load_progress(progress_path)
        if state.get("input_file") != str(input_path):
            raise RuntimeError("Progress file belongs to a different input file.")
        if state.get("chunk_chars") != chunk_chars:
            raise RuntimeError("Progress file chunk size does not match current --chunk-chars.")
        if state.get("skip_normalization") != skip_normalization:
            raise RuntimeError("Progress file skip-normalization setting does not match current run.")
        if state.get("target_lang_code") != target_lang_code:
            raise RuntimeError("Progress file target language code does not match current run.")
        if state.get("target_lang_name") != target_lang_name:
            raise RuntimeError("Progress file target language name does not match current run.")
        if state.get("total_chunks") != total:
            raise RuntimeError("Progress file chunk layout does not match the current source text.")
        normalized_chunks = list(state.get("normalized_chunks", []))
        english_chunks = list(state.get("english_chunks", []))
        start_idx = int(state.get("completed_chunks", 0))
        if start_idx != len(english_chunks):
            raise RuntimeError("Progress file is inconsistent: completed_chunks does not match english_chunks.")
        eprint(f"Resuming from chunk {start_idx + 1} of {total}...")
        if english_chunks:
            write_text_file(english_path, "\n\n".join(english_chunks).strip())
        if save_normalized and normalized_chunks and not skip_normalization:
            write_text_file(normalized_path, "\n\n".join(normalized_chunks).strip())

    for idx in range(start_idx, total):
        human_idx = idx + 1
        chunk = chunks[idx]

        if skip_normalization:
            normalized = chunk
            normalized_chunks.append(normalized)
            eprint(f"[{human_idx}/{total}] Skipping normalization; translating source text with {translator_model}...")
        else:
            eprint(f"[{human_idx}/{total}] Normalizing old Russian with {normalizer_model}...")
            normalized = normalize_old_russian(
                chunk,
                host,
                normalizer_model,
                temperature,
                keep_alive,
                timeout,
            )
            normalized_chunks.append(normalized)
            if save_normalized:
                write_text_file(normalized_path, "\n\n".join(normalized_chunks).strip())

            eprint(f"[{human_idx}/{total}] Translating to English with {translator_model}...")

        english = translate_russian_to_english(
            normalized,
            host,
            translator_model,
            temperature,
            keep_alive,
            timeout,
            target_lang_code,
            target_lang_name,
        )
        english_chunks.append(english)

        write_text_file(english_path, "\n\n".join(english_chunks).strip())
        save_progress(
            progress_path,
            input_file=str(input_path),
            chunk_chars=chunk_chars,
            total_chunks=total,
            skip_normalization=skip_normalization,
            target_lang_code=target_lang_code,
            target_lang_name=target_lang_name,
            completed_chunks=len(english_chunks),
            normalized_chunks=normalized_chunks,
            english_chunks=english_chunks,
        )

    normalized_text = "\n\n".join(normalized_chunks).strip()
    english_text = "\n\n".join(english_chunks).strip()

    if save_normalized and not skip_normalization:
        write_text_file(normalized_path, normalized_text)
        eprint(f"Wrote normalized Russian: {normalized_path}")
    else:
        normalized_path = None

    write_text_file(english_path, english_text)
    if progress_path.exists():
        progress_path.unlink()
    eprint(f"Wrote English translation: {english_path}")
    return english_path, normalized_path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Pavlov-era / pre-1918 Russian OCR text to a target language using Ollama."
    )
    parser.add_argument("input_file", help="Path to a source .txt file or a directory containing .txt files")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--normalizer-model",
        default=DEFAULT_NORMALIZER_MODEL,
        help=f"Model for old-Russian normalization (default: {DEFAULT_NORMALIZER_MODEL})",
    )
    parser.add_argument(
        "--translator-model",
        default=DEFAULT_TRANSLATOR_MODEL,
        help=f"Model for translation (default: {DEFAULT_TRANSLATOR_MODEL})",
    )
    parser.add_argument(
        "--target-lang-code",
        default=DEFAULT_TARGET_LANG_CODE,
        help=f"Target language code for translation (default: {DEFAULT_TARGET_LANG_CODE})",
    )
    parser.add_argument(
        "--target-lang-name",
        default=DEFAULT_TARGET_LANG_NAME,
        help=f"Target language name for translation prompts (default: {DEFAULT_TARGET_LANG_NAME})",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3000,
        help="Approximate max characters per chunk (default: 3000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for both model calls (default: 0.0)",
    )
    parser.add_argument(
        "--keep-alive",
        default="30m",
        help="How long Ollama keeps models loaded, e.g. 5m, 30m, 0 (default: 30m)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="HTTP timeout in seconds per request (default: 1800)",
    )
    parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Also save the normalized modern Russian text as <filename>_normalized_ru.txt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <filename>_progress.json if it exists",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip Qwen normalization and translate the source OCR text directly to English",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    input_path = pathlib.Path(args.input_file).expanduser().resolve()

    if not input_path.exists():
        eprint(f"Input path does not exist: {input_path}")
        return 2

    start = time.time()

    if input_path.is_dir():
        files = list_text_files(input_path)
        if not files:
            eprint(f"No .txt files found in: {input_path}")
            return 2

        translated_dir = input_path / "translated"
        translated_dir.mkdir(parents=True, exist_ok=True)
        eprint(f"Found {len(files)} .txt file(s) in {input_path}. Writing outputs to {translated_dir}")

        successes: List[tuple[pathlib.Path, pathlib.Path, pathlib.Path | None]] = []
        failures: List[tuple[pathlib.Path, str]] = []

        for file_idx, file_path in enumerate(files, start=1):
            eprint(f"[file {file_idx}/{len(files)}] Processing {file_path.name}...")
            try:
                english_path, normalized_path = process_file(
                    input_path=file_path,
                    host=args.host,
                    normalizer_model=args.normalizer_model,
                    translator_model=args.translator_model,
                    target_lang_code=args.target_lang_code,
                    target_lang_name=args.target_lang_name,
                    chunk_chars=args.chunk_chars,
                    temperature=args.temperature,
                    keep_alive=args.keep_alive,
                    timeout=args.timeout,
                    save_normalized=args.save_normalized,
                    skip_normalization=args.skip_normalization,
                    resume=args.resume,
                    output_dir=translated_dir,
                )
                successes.append((file_path, english_path, normalized_path))
            except KeyboardInterrupt:
                eprint("Interrupted by user.")
                eprint("Tip: rerun with --resume to continue from the last completed chunk.")
                return 130
            except Exception as exc:
                failures.append((file_path, str(exc)))
                eprint(f"Error in {file_path.name}: {exc}")
                continue

        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s")
        print(f"Translated directory: {translated_dir}")
        print(f"Succeeded: {len(successes)}")
        print(f"Failed: {len(failures)}")
        if failures:
            print("Failed files:")
            for file_path, message in failures:
                print(f"- {file_path.name}: {message}")
            return 1
        return 0

    if input_path.suffix.lower() != ".txt":
        eprint("Warning: input file is not .txt; continuing anyway.")

    try:
        english_path, normalized_path = process_file(
            input_path=input_path,
            host=args.host,
            normalizer_model=args.normalizer_model,
            translator_model=args.translator_model,
            target_lang_code=args.target_lang_code,
            target_lang_name=args.target_lang_name,
            chunk_chars=args.chunk_chars,
            temperature=args.temperature,
            keep_alive=args.keep_alive,
            timeout=args.timeout,
            save_normalized=args.save_normalized,
            skip_normalization=args.skip_normalization,
            resume=args.resume,
            output_dir=None,
        )
    except KeyboardInterrupt:
        eprint("Interrupted by user.")
        eprint("Tip: rerun with --resume to continue from the last completed chunk.")
        return 130
    except Exception as exc:
        eprint(f"Error: {exc}")
        eprint("Tip: use smaller --chunk-chars, a lighter --translator-model, and --resume on the next run.")
        return 1

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    print(f"Translated output: {english_path}")
    if normalized_path is not None:
        print(f"Normalized Russian: {normalized_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
