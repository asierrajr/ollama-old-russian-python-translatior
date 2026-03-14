# Old Russian OCR Translator for Ollama

Translate Pavlov-era / pre-1918 Russian OCR `.txt` files into a target language using Ollama.

This script can:

- translate a **single `.txt` file**
- translate **all top-level `.txt` files in a directory**
- optionally **normalize old/pre-1918 Russian** first with Qwen
- **resume** interrupted runs from a JSON progress file
- write outputs incrementally as each chunk completes
- detect and reject **summary / wrong-language** model output
- retry bad chunks with stricter prompts and smaller salvage sub-chunks

## Requirements

- Python 3.10+
- Ollama running locally, usually at `http://localhost:11434`
- Models pulled in Ollama, for example:
  - `qwen3:30b` for normalization
  - `translategemma:12b` for translation

Example:

```bash
ollama pull qwen3:30b
ollama pull translategemma:12b
```

## Quick start

Single file to English:

```bash
python old_russian_translate_patched_v5.py ./document.txt --target-lang-code en --target-lang-name English
```

Directory of `.txt` files to English:

```bash
python old_russian_translate_patched_v5.py ./ocr_texts --target-lang-code en --target-lang-name English
```

## What it outputs

### Single-file mode

For `document.txt` and target `en`, the script writes:

- `document_en.txt`
- `document_normalized_ru.txt` if `--save-normalized` is used and normalization is enabled
- `document_progress.json` while the file is in progress

### Directory mode

If you pass a directory like `./ocr_texts`, the script writes translated files into:

```text
./ocr_texts/translated/
```

Example output files:

- `./ocr_texts/translated/file1_en.txt`
- `./ocr_texts/translated/file2_en.txt`

If normalization is saved:

- `./ocr_texts/translated/file1_normalized_ru.txt`

## Progress / resume behavior

The script saves progress in:

```text
<filename>_progress.json
```

Use `--resume` to continue from the last completed chunk.

Important:

- progress is tracked **per source file**
- the JSON progress file is **deleted automatically after a successful run**
- if a run fails or is interrupted, the progress JSON remains so you can resume

## Arguments

### Positional

- `input_file`  
  Path to either:
  - a single source `.txt` file, or
  - a directory containing top-level `.txt` files

### Optional flags

- `--host`  
  Ollama host URL.  
  Default: `http://localhost:11434`

- `--normalizer-model`  
  Model used to normalize old/pre-1918 Russian before translation.  
  Default: `qwen3:30b`

- `--translator-model`  
  Model used for translation.  
  Default: `translategemma:12b`

- `--target-lang-code`  
  Target language code used for output naming and validation.  
  Default: `en`

- `--target-lang-name`  
  Human-readable target language name used in prompts.  
  Default: `English`

- `--chunk-chars`  
  Approximate maximum characters per chunk.  
  Default: `3000`

- `--temperature`  
  Sampling temperature for both model calls.  
  Default: `0.0`

- `--keep-alive`  
  How long Ollama keeps models loaded.  
  Examples: `5m`, `30m`, `0`  
  Default: `30m`

- `--timeout`  
  HTTP timeout in seconds per request.  
  Default: `1800`

- `--save-normalized`  
  Also save the normalized modern Russian text as:
  - `<filename>_normalized_ru.txt`

- `--resume`  
  Resume from `<filename>_progress.json` if it exists.

- `--skip-normalization`  
  Skip the Qwen normalization step and translate the source OCR text directly to the target language.

## Examples

### 1) Single file, English output

```bash
python old_russian_translate_patched_v5.py ./paper.txt --target-lang-code en --target-lang-name English
```

### 2) Single file, skip normalization

```bash
python old_russian_translate_patched_v5.py ./paper.txt --skip-normalization --target-lang-code en --target-lang-name English
```

### 3) Single file, save normalized Russian too

```bash
python old_russian_translate_patched_v5.py ./paper.txt --save-normalized --target-lang-code en --target-lang-name English
```

### 4) Resume an interrupted run

```bash
python old_russian_translate_patched_v5.py ./paper.txt --resume --target-lang-code en --target-lang-name English
```

### 5) Use smaller chunks for messy OCR

```bash
python old_russian_translate_patched_v5.py ./paper.txt --chunk-chars 1500 --target-lang-code en --target-lang-name English
```

### 6) Use a lighter translator model

```bash
python old_russian_translate_patched_v5.py ./paper.txt --translator-model translategemma:4b --target-lang-code en --target-lang-name English
```

### 7) Translate to French

```bash
python old_russian_translate_patched_v5.py ./paper.txt --target-lang-code fr --target-lang-name French
```

Output:

```text
paper_fr.txt
```

### 8) Translate to Brazilian Portuguese

```bash
python old_russian_translate_patched_v5.py ./paper.txt --target-lang-code pt-BR --target-lang-name "Brazilian Portuguese"
```

Output:

```text
paper_pt-BR.txt
```

### 9) Translate all `.txt` files in a directory

```bash
python old_russian_translate_patched_v5.py ./ocr_texts --target-lang-code en --target-lang-name English
```

Outputs go to:

```text
./ocr_texts/translated/
```

### 10) Directory mode with resume and skip-normalization

```bash
python old_russian_translate_patched_v5.py ./ocr_texts --resume --skip-normalization --target-lang-code en --target-lang-name English
```

## Notes

- Directory mode scans **top-level `*.txt` files only**. It does not recurse into subfolders.
- In directory mode, if one file fails, the script continues with the rest and prints a final success/failure summary.
- Resume works only when key settings match the original run, including:
  - `--chunk-chars`
  - `--skip-normalization`
  - `--target-lang-code`
  - `--target-lang-name`
- If you change those settings, start fresh or delete the old progress JSON first.
- If the model returns a summary or wrong-language output, the script retries that chunk automatically with stricter prompts and smaller salvage sub-chunks.

## Suggested commands

Fastest / lighter run on a laptop:

```bash
python old_russian_translate_patched_v5.py ./paper.txt --skip-normalization --translator-model translategemma:4b --chunk-chars 1500 --timeout 7200 --target-lang-code en --target-lang-name English
```

Higher-quality run:

```bash
python old_russian_translate_patched_v5.py ./paper.txt --translator-model translategemma:12b --chunk-chars 3000 --timeout 7200 --target-lang-code en --target-lang-name English
```
