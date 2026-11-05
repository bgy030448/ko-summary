r"""한국어 요약 화면. 실행: .venv\Scripts\python.exe -m streamlit run app.py"""
import base64
import hashlib
from html import escape, unescape
from io import BytesIO
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import zipfile

import streamlit as st
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

MODEL_RELEASE = ROOT / "artifacts/model_release.json"
TOKENIZER_DIR = ROOT / "artifacts/kobart_base_v2"

LINK = re.compile(r'!?\[([^\]\n]*)\]\(\s*(https?://(?:[^\s()]|\([^()]*\))+)(?:\s+"[^"\n]*")?\s*\)')
URL = re.compile(r'https?://[^\s<>]+', re.I)
CAPTION = re.compile(r'(?:/|ⓒ|©)\s*(?:사진\s*[=:：]?\s*)?(?:뉴스1|연합뉴스|뉴시스|[^/\n]{1,25}(?:뉴스|일보|신문|기자))\s*$')
NOTICE = re.compile(r'^(?:광고|후원받음|Sponsored|Advertisement|무단\s*전재.*|저작권자.*|관련\s*기사|추천\s*기사|기사\s*공유|구독하기|좋아요)$', re.I)

SELECTOR = ROOT / "artifacts/rewrite_selector.pt"
MANIFEST = ROOT / "artifacts/rewrite_release.json"
DECODER = ROOT / "artifacts/rewrite_decoder.safetensors"
GENERATOR_MANIFEST = ROOT / "artifacts/rewrite_generator.json"
MAX_DOCUMENT_TOKENS = 4096
MAX_EVIDENCE_TOKENS = 480
GENERATION = {"max_new_tokens": 112, "min_new_tokens": 8, "num_beams": 4,
              "do_sample": False, "no_repeat_ngram_size": 3, "length_penalty": 1.1}

FACTUAL_DIRECTORY = ROOT / 'artifacts/factual'
FACTUAL_RELEASE = FACTUAL_DIRECTORY / 'service_release.json'
NLI_DIRECTORY = ROOT / 'artifacts/klue_nli'
CANDIDATE_SETTINGS = {**GENERATION, 'num_return_sequences': 4,
            'return_dict_in_generate': True, 'output_scores': True}

# 1. 붙여 넣은 뉴스와 첨부 문서 읽기

def preprocess_news(text: str) -> str:
    """구조가 남아 있을 때 잡음을 제거하고, 문장부호를 보존해 한 문단으로 반환합니다.

    이미지/추적 링크의 문구와 똑같은 독립 행도 제거합니다.
    '광고'라는 단어가 본문에 등장한다는 이유로 문장을 삭제하지 않습니다.
    """
    text = unicodedata.normalize('NFC', unescape(text))
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\u200b\ufeff]', '', text)
    discarded_labels = set()

    def label_key(label):
        return re.sub(r'[*_`]', '', label).strip()

    def replace_link(match):
        label, address = match.group(1), match.group(2).replace('\\&', '&')
        parsed = urlsplit(address)
        host, path = (parsed.hostname or '').lower(), parsed.path.lower()
        image = (match.group(0).startswith('!') or host.startswith('img-s-msn-com.')
                 or re.search(r'\.(?:png|jpe?g|gif|webp|svg|img)$', path))
        tracking = (host in {'bing.com', 'www.bing.com'}
                    and path.startswith(('/api/v2/mediation/', '/aclick')))
        if image or tracking:
            discarded_labels.add(label_key(label))
            return ''
        return label  # 본문의 출처 링크 등은 연결 문구를 살립니다.

    # 후원 표지가 달린 독립 행은 광고주 이름까지 함께 제거합니다.
    text = '\n'.join(
        '' if any(label_key(m.group(1)) in {'후원받음', 'Sponsored', 'Advertisement'}
                  for m in LINK.finditer(line)) else line
        for line in text.splitlines()
    )
    text = LINK.sub(replace_link, text)
    kept = []
    for line in text.splitlines():
        line = URL.sub('', line)
        line = re.sub(r'[*_`]', '', line).strip()
        line = re.sub(r'\s+', ' ', line)
        if not line or line in discarded_labels:
            continue
        if NOTICE.fullmatch(line) or CAPTION.search(line):
            continue
        # 이미지 링크만 남았던 괄호 등 독립적인 기호 행
        if not re.search(r'[가-힣A-Za-z0-9]', line):
            continue
        kept.append(line)
    return ' '.join(kept).replace('⁓', '~').replace('⁚', ':')

def read_document(name, content):
    """첨부 문서의 텍스트만 메모리에서 읽는다. 원문을 서버 파일로 남기지 않는다."""
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("문서 파일은 10MB 이하로 올려 주세요.")
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        for encoding in ("utf-8-sig", "cp949"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("텍스트 파일을 UTF-8 형식으로 저장해 주세요.")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("암호를 해제한 PDF를 올려 주세요.")
        if len(reader.pages) > 30:
            raise ValueError("30쪽 이하의 문서를 올려 주세요.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        with zipfile.ZipFile(BytesIO(content)) as archive:
            item = archive.getinfo("word/document.xml")
            if item.file_size > 20 * 1024 * 1024:
                raise ValueError("문서의 텍스트 용량이 너무 큽니다.")
            root = ET.fromstring(archive.read(item))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text = "\n".join("".join(p.itertext()) for p in root.findall(".//w:p", ns))
    else:
        raise ValueError("PDF, DOCX 또는 TXT 파일을 올려 주세요.")
    if len(text.strip()) < 30:
        raise ValueError("읽을 수 있는 본문이 부족합니다. 스캔 이미지는 지원하지 않으므로 텍스트를 붙여 넣어 주세요.")
    if len(text) > 60000:
        raise ValueError("문서가 너무 깁니다. 한 주제의 본문을 나눠 입력해 주세요.")
    return text.strip()

# 2. 저장된 모델로 요약 생성하기

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.replace("⁓", "~").replace("⁚", ":")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_release():
    if not MODEL_RELEASE.exists():
        return None
    return json.loads(MODEL_RELEASE.read_text(encoding="utf-8"))

def load_tokenizer():
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def input_ids(text, tokenizer):
    normalized = normalize_text(text)
    ids = tokenizer.encode(normalized, add_special_tokens=False)
    return [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]


def load_model(model_directory, expected_sha256=None):
    import torch
    from transformers import BartForConditionalGeneration
    directory = Path(model_directory).resolve()
    assert directory.is_relative_to(ROOT / "artifacts"), "프로젝트에서 확인한 모델만 불러옵니다."
    if expected_sha256:
        assert sha256(directory / "model.safetensors") == expected_sha256, "모델 파일 식별값이 다릅니다."
    torch.set_num_threads(4)
    model = BartForConditionalGeneration.from_pretrained(directory, local_files_only=True,
                use_safetensors=True, attn_implementation="sdpa")
    model.eval()
    return model

def generate(text, model, tokenizer, settings, max_source_tokens=576):
    import torch
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("요약할 원문을 입력해 주세요.")
    ids = input_ids(normalized, tokenizer)
    if len(ids) > max_source_tokens:
        raise ValueError(f"현재 원문은 {len(ids)}토큰입니다. {max_source_tokens}토큰 이하의 문단을 입력해 주세요. 원문을 자동으로 자르지 않습니다.")
    start = time.perf_counter()
    data = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(input_ids=data, attention_mask=torch.ones_like(data),
                                use_cache=True, **settings)
    summary = tokenizer.decode(output[0], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).strip()
    return {"source": normalized, "summary": summary, "source_tokens": len(ids),
            "seconds": time.perf_counter()-start, **review_signals(normalized, summary)}


def number_tokens(text):
    return {number.replace(",", "") for number in re.findall(r"[0-9]+(?:[.,][0-9]+)*", text)}


def char_lcs_f1(prediction, reference):
    a, b = re.sub(r"\s+", "", prediction), re.sub(r"\s+", "", reference)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b)+1)
    for char in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j-1]+1 if char == other else max(previous[j], current[-1]))
        previous = current
    return 2*previous[-1]/(len(a)+len(b))


def sentence_candidates(source):
    # 소수점 내부를 문장 끝으로 나누지 않습니다.
    return [s.strip() for s in re.split(r"(?<=[.!?。])\s+", normalize_text(source)) if s.strip()]


def review_signals(source, summary):
    words = summary.split()
    triples = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    related = sorted(enumerate(sentence_candidates(source)),
                     key=lambda pair: (-char_lcs_f1(summary, pair[1]), pair[0]))[:2]
    return {"numbers_not_in_source": sorted(number_tokens(summary)-number_tokens(source)),
            "source_number_tokens": sorted(number_tokens(source)),
            "repeated_word_trigrams": len(triples)-len(set(triples)),
            "related_source_sentences": [text for _, text in related]}

def sentence_units(text, tokenizer):
    """문장 끝과 문단을 이용해 나눈다. 너무 긴 한 문장도 버리지 않고 토큰 구간으로 나눈다."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize_text(paragraph)
        # 따옴표 자체는 문장 끝이 아니다. ‘제품명’ 다음에서도 문맥을 유지한다.
        parts = re.split(r'(?<=[.!?。])\s+|(?<=[.!?。][”’\"])\s+', paragraph)
        for part in parts:
            if not part.strip():
                continue
            encoded = tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            if not offsets:
                continue
            if len(offsets) <= 510:
                units.append(part)
            else:
                start = 0
                for end in range(510, len(offsets) + 510, 510):
                    boundary = offsets[min(end, len(offsets)) - 1][1]
                    units.append(part[start:boundary].strip())
                    start = boundary
    return [u for u in units if u]


def document_features(text, model, tokenizer):
    count = len(input_ids(text, tokenizer))
    if not normalize_text(text):
        raise ValueError("요약할 본문을 입력해 주세요.")
    if count > MAX_DOCUMENT_TOKENS:
        raise ValueError(f"현재 문서는 {count}토큰입니다. 한 주제의 본문을 {MAX_DOCUMENT_TOKENS}토큰 이하로 나눠 주세요. 자동으로 자르지 않습니다.")
    units = sentence_units(text, tokenizer)
    if not units:
        raise ValueError("읽을 수 있는 문장을 찾지 못했습니다.")
    groups, current = [], []
    for i, unit in enumerate(units):
        if current and len(input_ids(" ".join(units[j] for j in [*current, i]), tokenizer)) > 576:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    vectors = [None] * len(units)
    model.eval()
    with torch.inference_mode():
        for group in groups:
            joined = " ".join(units[j] for j in group)
            encoded = tokenizer(joined, add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor([[tokenizer.bos_token_id, *encoded["input_ids"], tokenizer.eos_token_id]],
                               device=next(model.parameters()).device)
            hidden = model.get_encoder()(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
            cursor = 0
            for j in group:
                end = cursor + len(units[j])
                indices = [k + 1 for k, (a, b) in enumerate(encoded["offset_mapping"]) if b > cursor and a < end]
                if not indices:
                    raise ValueError("문장의 토큰 범위를 찾지 못했습니다.")
                vectors[j] = hidden[indices].mean(0).cpu()
                cursor = end + 1
    embeddings = torch.stack(vectors).clone()
    # 문장 벡터는 같은 구간의 앞뒤 문맥을 포함한다. 문서 평균과 상대 위치도 함께 사용한다.
    document = embeddings.mean(0).expand_as(embeddings)
    metadata = torch.tensor([[i / max(1, len(units) - 1), min(len(u), 500) / 500]
                             for i, u in enumerate(units)], dtype=torch.float32)
    features = torch.cat([embeddings, document, metadata], dim=1)
    return units, features, count, len(groups)

class SentenceSelector(nn.Module):
    def __init__(self, hidden=768):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(hidden * 2 + 2), nn.Linear(hidden * 2 + 2, 96),
                                    nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def load_selector():
    if not SELECTOR.exists() or not MANIFEST.exists():
        raise ValueError("핵심 문장 선별 모델을 먼저 준비해 주세요.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256(SELECTOR) != manifest["selector_sha256"]:
        raise ValueError("선별 모델 파일이 준비 기록과 다릅니다.")
    if sha256(ROOT / "artifacts/model_release.json") != manifest["base_release_sha256"]:
        raise ValueError("생성 모델이 선별 모델 학습 당시와 다릅니다.")
    head = SentenceSelector()
    head.load_state_dict(torch.load(SELECTOR, map_location="cpu", weights_only=True))
    return head.eval()


def load_rewriter():
    """기존 가중치를 별도로 불러와 학습한 디코더 두 층만 교체한다."""
    from safetensors.torch import load_file
    release = read_release()
    model = load_model(ROOT / release["model_directory"], release["weights_sha256"])
    if GENERATOR_MANIFEST.exists():
        manifest = json.loads(GENERATOR_MANIFEST.read_text(encoding="utf-8"))
        if manifest["base_release_sha256"] != sha256(ROOT / "artifacts/model_release.json"):
            raise ValueError("추가 학습의 원본 모델이 다릅니다.")
        if manifest["selector_manifest_sha256"] != sha256(MANIFEST) or manifest["weights_sha256"] != sha256(DECODER):
            raise ValueError("추가 학습 모델의 식별값이 다릅니다.")
        state = load_file(str(DECODER))
        expected = {name for name, _ in model.named_parameters() if decoder_parameter(name)}
        if set(state) != expected:
            raise ValueError("추가 학습한 디코더의 구성과 다릅니다.")
        model.load_state_dict(state, strict=False)
    return model.eval()


def decoder_parameter(name):
    return name.startswith(("model.decoder.layers.4.", "model.decoder.layers.5."))

def select_evidence(units, scores, tokenizer):
    selected = []
    # 중요도 순으로 예산을 배분한 뒤 원문 순서로 복원한다. 목록을 그대로 결과로 반환하지 않는다.
    for index in sorted(range(len(units)), key=lambda i: (-scores[i], i)):
        proposal = sorted([*selected, index])
        if len(input_ids(" ".join(units[i] for i in proposal), tokenizer)) <= MAX_EVIDENCE_TOKENS:
            selected = proposal
        if len(selected) >= 5:
            break
    if not selected:
        # 한 문장 자체가 예산보다 길면 모델 허용 범위 안에서 그 문장을 온전히 사용한다.
        selected = [max(range(len(units)), key=lambda i: scores[i])]
    return selected, " ".join(units[i] for i in selected)


def rewriting_signals(source, summary):
    compact = lambda value: re.sub(r"\W+", "", value)
    source_key, summary_key = compact(source), compact(summary)
    source_grams = {source_key[i:i+6] for i in range(max(0, len(source_key)-5))}
    grams = [summary_key[i:i+6] for i in range(max(0, len(summary_key)-5))]
    words = summary.split()
    adjacent = sorted({a for a, b in zip(words, words[1:]) if a == b})
    return {"verbatim_source_span": bool(summary_key and summary_key in source_key),
            "adjacent_repeated_words": adjacent,
            "novel_char_6gram_ratio": sum(g not in source_grams for g in grams) / len(grams) if grams else 0.,
            "compression_ratio": len(summary_key) / max(1, len(source_key))}


def summarize_document(text, model, tokenizer, selector):
    start = time.perf_counter()
    units, features, count, chunks = document_features(text, model, tokenizer)
    with torch.inference_mode():
        scores = selector(features).sigmoid().tolist()
    indices, evidence = select_evidence(units, scores, tokenizer)
    result = generate(evidence, model, tokenizer, GENERATION, max_source_tokens=576)
    result.update({"source": normalize_text(text), "source_tokens": count,
                   "evidence_tokens": len(input_ids(evidence, tokenizer)),
                   "selected_sentences": [units[i] for i in indices],
                   "selected_indices": indices, "sentence_count": len(units), "chunks": chunks,
                   "pipeline": "contextual_sentence_selector_then_kobart_generator",
                   "seconds": time.perf_counter() - start})
    result.update(review_signals(text, result["summary"]))
    result.update(rewriting_signals(text, result["summary"]))
    result["numbers_not_in_evidence"] = review_signals(evidence, result["summary"])["numbers_not_in_source"]
    return result

# 3. 주체·근거·반복을 대조해 후보 고르기

def subject_anchor(text):
    text = re.sub(r'^(?:또한|아울러|한편|이에|특히)[, ]+', '', text.strip())
    match = re.match(r'^[\s“‘\"\'(]*([가-힣A-Za-z0-9&·]{2,30}?)(?:에서는|에서|은|는|이|가|의)(?=[\s‘“\"\'])', text)
    return match.group(1) if match else None


def anchored_premise(source, summary, strict_names=True):
    """명시적인 첫 주체가 있는 경우 해당 주체의 원문과 바로 이어진 지시문만 대조한다."""
    anchor = subject_anchor(summary)
    if not anchor:
        return source, None
    sentences = re.split(r'(?<=[.!?。])\s+', normalize_text(source))
    # '카카오'와 '카카오페이'처럼 이름 일부만 같은 다른 주체를 연결하지 않는다.
    name_pattern = re.compile(r'(?<![\w])'+re.escape(anchor)+r'(?=$|[\s.,“‘”’\"\'()]|(?:은|는|이|가|의|와|과|에서|에게|을|를|도|로)(?=\s|[.,]))')
    chosen = [i for i,s in enumerate(sentences) if (name_pattern.search(s) if strict_names else anchor in s)]
    if not chosen:
        return source, None
    for i in list(chosen):
        if i+1 < len(sentences) and re.match(r'^(?:이 회사|해당 회사|이 제품|해당 제품|이 서비스|이를 통해|이에 따라)', sentences[i+1]):
            chosen.append(i+1)
    return ' '.join(sentences[i] for i in sorted(set(chosen))), anchor


def adjacent_words(text):
    return [re.sub(r'^[\W_]+|[\W_]+$', '',w) for w in text.split()]


def punctuation_repetition(text):
    words=adjacent_words(text)
    return sorted({a for a,b in zip(words,words[1:]) if a and a==b})


def multiple_subject_risk(evidence, summary):
    anchor=subject_anchor(summary)
    if not anchor:
        return False
    generic={'이들','그들','우리','기존','홈페','정부','기업','회사','제품','서비스','소비자','고객',
             '전문가들','전문','올해','내년','지난해','국민','당국','업체','이것','그것'}
    if anchor in generic:
        return False
    subjects={subject_anchor(s) for s in re.split(r'(?<=[.!?。])\s+',normalize_text(evidence))}
    subjects.discard(None)
    return bool(subjects-generic-{anchor})


def choose_candidate(evidence,candidates):
    """정상 1순위는 보존한다. NLI 점수 상승 자체를 재작성 이유로 삼지 않는다."""
    for candidate in candidates:
        repeated=bool(punctuation_repetition(candidate['summary']) or candidate['repeated_word_trigrams'])
        candidate['punctuation_insensitive_repeated_words']=punctuation_repetition(candidate['summary'])
        candidate['ranking_score']=(candidate['support']-candidate['contradiction']-float(repeated)
            -float(bool(candidate['numbers_not_in_evidence']))+.05*candidate['generation_score'])
    first=candidates[0]
    risk=multiple_subject_risk(evidence,first['summary'])
    if (not first['punctuation_insensitive_repeated_words'] and not first['repeated_word_trigrams']
        and not first['numbers_not_in_evidence'] and (not risk or first['support']>=.5)):
        return 0
    return max(range(len(candidates)),key=lambda i:candidates[i]['ranking_score'])

class SupportScorer:
    """원문 구간이 생성문을 뒷받침하는지 예측. 점수는 사실 정확도가 아니다."""
    def __init__(self, device='cpu', scope_mode='subject_local_v2'):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_DIRECTORY, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_DIRECTORY, local_files_only=True, trust_remote_code=False).to(device).eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if not {'entailment', 'contradiction'} <= labels.keys():
            raise RuntimeError(f'NLI 라벨을 확인할 수 없습니다: {labels}')
        self.entailment, self.contradiction = labels['entailment'], labels['contradiction']
        self.device = device
        if scope_mode not in ['global_and_subject_v1','subject_local_v2']:
            raise ValueError('알 수 없는 관계 대조 버전입니다.')
        self.scope_mode = scope_mode

    def _score_windows(self, source, summaries):
        # 처음 512토큰만 잘라 버리지 않는다. 겹치는 모든 구간을 검사한다.
        ids = self.tokenizer.encode(normalize_text(source), add_special_tokens=False)
        windows = [self.tokenizer.decode(ids[i:i+320], skip_special_tokens=True)
                   for i in range(0, len(ids), 224)]
        if not windows:
            windows = ['']
        pairs = [(w, s) for s in summaries for w in windows]
        values = []
        with torch.inference_mode():
            for offset in range(0, len(pairs), 4):
                batch = pairs[offset:offset+4]
                encoded = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                    padding=True, truncation='only_first', max_length=510, return_tensors='pt').to(self.device)
                # 이 공개 체크포인트는 BertTokenizer + type_vocab_size=1 RoBERTa다.
                # 두 번째 문장의 segment=1을 전달하면 임베딩 범위를 벗어난다.
                encoded.pop('token_type_ids', None)
                values.extend(self.model(**encoded).logits.softmax(-1).cpu().tolist())
        scores = []
        for i in range(len(summaries)):
            chunk = values[i*len(windows):(i+1)*len(windows)]
            best = max(chunk, key=lambda p: p[self.entailment]-p[self.contradiction])
            scores.append({'support': best[self.entailment], 'contradiction': best[self.contradiction]})
        return scores

    def score(self, source, summaries):
        global_scores = self._score_windows(source, summaries)
        for summary, score in zip(summaries, global_scores):
            # 같은 원문에 나온다는 것만으로 기업과 속성의 연결을 인정하지 않는다.
            # 명시적 주체가 없는 문장/복잡한 공지시 등은 이 규칙으로 보장할 수 없다.
            claims = re.split(r'(?<=[.!?。])\s+', summary)
            anchors, scoped = [], []
            for claim in claims:
                premise, anchor = anchored_premise(source, claim,strict_names=self.scope_mode=='subject_local_v2')
                if anchor:
                    anchors.append(anchor)
                    scoped.append(self._score_windows(premise, [claim])[0])
            score['global_support'] = score['support']
            score['subject_anchors'] = anchors
            if scoped:
                # 긴 원문 전체는 명백히 맞는 인용도 중립으로 오판할 수 있다.
                # 명시적 주체가 있는 주장은 해당 주체의 근거로 판단한다.
                score['support'] = min([s['support'] for s in scoped]+([score['support']] if self.scope_mode=='global_and_subject_v1' else []))
                score['contradiction'] = max([s['contradiction'] for s in scoped]+([score['contradiction']] if self.scope_mode=='global_and_subject_v1' else []))
        return global_scores

def generate_candidates(evidence, model, tokenizer):
    ids = input_ids(evidence, tokenizer)
    if len(ids) > 576:
        raise ValueError('선별 근거가 입력 범위를 넘었습니다.')
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                use_cache=True, **CANDIDATE_SETTINGS)
    return [{'summary': tokenizer.decode(seq, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).strip(),
             'generation_score': float(score)}
            for seq, score in zip(output.sequences, output.sequences_scores)]


def rank_candidates(source, evidence, candidates, scorer, conservative=None):
    support = scorer.score(source, [c['summary'] for c in candidates])
    enriched = []
    for candidate, nli in zip(candidates, support):
        text = candidate['summary']
        signals = {**review_signals(source, text), **rewriting_signals(source, text)}
        unsupported_numbers = review_signals(evidence, text)['numbers_not_in_source']
        repetition = bool(signals['adjacent_repeated_words'] or signals['repeated_word_trigrams'])
        # 이름별 예외/결과 문자열 수정은 없다. 모든 문서에 동일한 후보 선택 규칙.
        rank = (nli['support'] - nli['contradiction'] - 1.0*repetition
                - 1.0*bool(unsupported_numbers) + .05*candidate['generation_score'])
        enriched.append({**candidate, **signals, **nli, 'ranking_score': rank,
                         'numbers_not_in_evidence': unsupported_numbers})
    if conservative is None:
        conservative=getattr(scorer,'scope_mode','subject_local_v2')!='global_and_subject_v1'
    best = choose_candidate(evidence,enriched) if conservative else max(range(len(enriched)),key=lambda i:enriched[i]['ranking_score'])
    return enriched, best

def repair_candidates(source, evidence, assessed, best, model, tokenizer, scorer, conservative=True):
    """모든 후보의 근거가 약하면 해당 주체의 근거만으로 한 번 더 생성한다.

    같은 규칙/최대 1회/4개 후보를 전후 모두 적용한다. 잘못된 문구를 문자열 치환하지 않는다.
    """
    first=assessed[0]
    if conservative and (not multiple_subject_risk(evidence,first['summary'])
        and not punctuation_repetition(first['summary']) and not first['repeated_word_trigrams']
        and not first['numbers_not_in_evidence']):
        return assessed,best,None
    if any(c['support'] >= .5 and not (punctuation_repetition(c['summary']) if conservative else c['adjacent_repeated_words'])
           and not c['repeated_word_trigrams']
           and not c['numbers_not_in_evidence'] for c in assessed):
        return assessed, best, None
    # 선별 과정에서 주체의 소개 문장이 빠졌다면 원문에서 함께 복원한다.
    # 기존 비교 정책은 선택된 근거만 사용하며, 새 정책은 양쪽 모델 모두 원문을 사용한다.
    premise, anchor = anchored_premise(source if conservative else evidence, assessed[best]['summary'])
    if len(input_ids(premise,tokenizer)) > 576:
        premise,anchor=anchored_premise(evidence,assessed[best]['summary'])
    if not anchor or normalize_text(premise) == normalize_text(evidence):
        return assessed, best, None
    if not premise.strip() or len(input_ids(premise,tokenizer)) > 576:
        return assessed, best, None
    proposed = generate_candidates(premise,model,tokenizer)
    repaired, _ = rank_candidates(source,premise,proposed,scorer,conservative=conservative)
    for c in repaired:
        c['repair_evidence'] = premise
    merged = [*assessed,*repaired]
    choice = choose_candidate(evidence,merged) if conservative else max(range(len(merged)),key=lambda i:merged[i]['ranking_score'])
    return merged, choice, {'anchor':anchor,'premise':premise,'generated_candidates':len(repaired)}

def clean_accidental_repetition(source, summary):
    """원문에 없는 인접 동일 단어만 정리한다. 원래 생성문과 편집 내역은 보존한다."""
    source_words=adjacent_words(normalize_text(source))
    source_pairs=set(zip(source_words,source_words[1:]))
    words, edits = [], []
    for word in summary.split():
        key=adjacent_words(word)[0]
        previous=adjacent_words(words[-1])[0] if words else None
        if (words and key == previous and len(key)>1 and not key.isnumeric()
                and (key,key) not in source_pairs):
            edits.append(key)
            words[-1]=word
        else:
            words.append(word)
    return ' '.join(words), edits


def finish_candidate(source, candidate, scorer):
    result = dict(candidate)
    cleaned, edits = clean_accidental_repetition(source,result['summary'])
    result['raw_generated_summary'] = result['summary']
    result['repetition_cleanup'] = edits
    if edits:
        result['summary'] = cleaned
        result.update(review_signals(source,cleaned))
        result.update(rewriting_signals(source,cleaned))
        result.update(scorer.score(source,[cleaned])[0])
    result['punctuation_insensitive_repeated_words']=punctuation_repetition(result['summary'])
    return result

def load_factual_model():
    from safetensors.torch import load_file

    release = json.loads(FACTUAL_RELEASE.read_text(encoding='utf-8'))
    if sha256(GENERATOR_MANIFEST) != release['initial_generator_manifest_sha256']:
        raise ValueError('비교 시작 모델이 학습 기록과 다릅니다.')
    path = FACTUAL_DIRECTORY / release['checkpoint']
    if sha256(path) != release['weights_sha256']:
        raise ValueError('개선 모델의 식별값이 다릅니다.')
    model = load_rewriter()
    if release.get('use_initial_model', False):
        return model.eval()
    state = load_file(str(path))
    if set(state) != {n for n, _ in model.named_parameters() if decoder_parameter(n)}:
        raise ValueError('개선 모델의 층 구성이 다릅니다.')
    model.load_state_dict(state, strict=False)
    return model.eval()


def summarize_factual(text, model, tokenizer, selector, scorer):
    start = time.perf_counter()
    units, features, count, chunks = document_features(text, model, tokenizer)
    with torch.inference_mode():
        scores = selector(features).sigmoid().tolist()
    indices, evidence = select_evidence(units, scores, tokenizer)
    candidates = generate_candidates(evidence, model, tokenizer)
    assessed, best = rank_candidates(text, evidence, candidates, scorer)
    assessed, best, repair = repair_candidates(text,evidence,assessed,best,model,tokenizer,scorer)
    result = finish_candidate(text,assessed[best],scorer)
    used_evidence = result.get('repair_evidence',evidence)
    result.update({'source': normalize_text(text), 'source_tokens': count,
                   'selected_sentences': re.split(r'(?<=[.!?。])\s+',used_evidence),
                   'initial_selected_sentences': [units[i] for i in indices],
                   'selected_indices': [i for i in range(len(units)) if units[i] in used_evidence],
                   'initial_selected_indices': indices, 'sentence_count': len(units), 'chunks': chunks,
                   'evidence_tokens': len(input_ids(used_evidence, tokenizer)),
                   'seconds': time.perf_counter()-start, 'candidate_index': best,
                   'pipeline': 'trained_rewriter_with_support_reranking',
                   'repair_used': 'repair_evidence' in result, 'repair_attempted':repair is not None,
                   'requires_review': result['support'] < .5 or result['contradiction'] > .25,
                   'support_limit': '한국어 문장 관계 모델의 보조 신호이며 사실 보증이 아닙니다.'})
    return result

# 4. 화면과 입력·출력 처리

def main():
    st.set_page_config(page_title="한국어 핵심 요약", page_icon="📄", layout="wide")


    @st.cache_data(show_spinner=False)
    def browser_font_css():
        """설치된 글꼴을 브라우저에도 전달해 다른 글꼴로 대체되지 않게 한다."""
        font_dir = Path.home() / "AppData/Local/Microsoft/Windows/Fonts"
        rules = []
        for name, weight in (("Regular", 400), ("Bold", 700)):
            font_path = font_dir / f"Pretendard-{name}.otf"
            if font_path.exists():
                encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
                rules.append("@font-face {font-family:Pretendard;src:url(data:font/otf;base64,"
                             + encoded + ") format('opentype');font-weight:" + str(weight)
                             + ";font-style:normal;font-display:swap;}")
        return "\n".join(rules)


    st.markdown("<style>" + browser_font_css() + """
    :root {--paper:#FAF9F7;--sand:#F0ECE7;--ink:#29282D;--muted:#736D68;--line:#D8D1CA;--accent:#466878;}
    .stApp, .stApp p, .stApp input, .stApp textarea, .stApp button, .stApp label,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] div {
      font-family:Pretendard,"Malgun Gothic",sans-serif;}
    .stApp {background:var(--paper);color:var(--ink);}
    header[data-testid="stHeader"] {background:var(--paper);}
    .block-container {max-width:1360px;padding:4.5rem 3rem 2.5rem;}
    h1,h2,h3 {font-family:Pretendard,"Malgun Gothic",sans-serif!important;color:var(--ink);letter-spacing:-.04em;}
    h1 {font-size:3.2rem!important;font-weight:700!important;line-height:1.18!important;padding:.45rem 0 1rem!important;}
    h2,h3 {font-size:1.45rem!important;line-height:1.3!important;}
    .editorial-hero {padding:0 0 1.5rem;border-bottom:1px solid var(--line);margin-bottom:1.8rem;}
    .eyebrow {color:var(--muted);font-size:.78rem;letter-spacing:.12em;margin-bottom:.7rem;}
    .hero-description {color:var(--muted);font-size:1.05rem;line-height:1.8;max-width:680px;margin:0;}
    .section-number {font-size:.78rem;color:var(--muted);letter-spacing:.12em;margin:0 0 -.4rem;}
    .section-rule {width:48px;height:2px;background:#9F9289;margin:.35rem 0 1rem;}
    div[data-testid="stTextArea"] textarea {line-height:1.85;background:#F0ECE7!important;color:var(--ink)!important;padding:1.1rem!important;border-radius:0!important;}
    div[data-testid="stTextArea"] > div, div[data-baseweb="select"] > div {border-radius:0!important;border-color:var(--line)!important;}
    div[data-testid="stTextArea"] textarea:focus-visible {outline:2px solid var(--accent);outline-offset:-2px;}
    div[data-testid="stSelectbox"] [data-baseweb="select"] {font-size:.95rem;}
    .stApp button {border-radius:2px!important;box-shadow:none!important;min-height:2.8rem;}
    button[kind="primary"] {background:#34343A!important;border-color:#34343A!important;color:white!important;}
    button[kind="primary"]:hover {background:#466878!important;border-color:#466878!important;}
    button:focus-visible {outline:2px solid var(--accent)!important;outline-offset:3px;}
    div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {color:var(--muted)!important;}
    .summary-paper {background:#EAE4DD;padding:1.6rem 1.8rem;margin:0 0 .5rem;border-top:2px solid #9F9289;}
    .summary-label {font-size:.75rem;color:#665F59;letter-spacing:.12em;margin-bottom:.7rem;}
    .summary-copy {font-size:1.2rem;line-height:1.85;letter-spacing:-.02em;color:var(--ink);overflow-wrap:anywhere;}
    .source-excerpt {border-left:2px solid #94A6AE;padding:.35rem 0 .35rem 1rem;margin:.5rem 0;color:#4D565B;line-height:1.8;font-size:.94rem;overflow-wrap:anywhere;}
    .empty-paper {background:var(--sand);min-height:330px;padding:2.2rem;border-top:2px solid #B8ADA4;}
    .empty-paper .empty-title {font-size:1.45rem;line-height:1.6;margin:1.5rem 0 .7rem;font-weight:700;}
    .empty-paper p {color:var(--muted);font-size:.95rem;line-height:1.9;max-width:340px;}
    .stApp hr {border-color:var(--line);margin:1.8rem 0 1.3rem;}
    div[data-testid="stExpander"] details {border-color:var(--line);border-radius:0;}
    @media(max-width:760px) {
     .block-container {padding:3.5rem 1.3rem 2rem;}
     h1 {font-size:2.35rem!important;}
     .summary-paper,.empty-paper {padding:1.35rem;}
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="editorial-hero">
      <div class="eyebrow">마케팅 · IT 기획을 위한 문서 읽기</div>
      <h1>한국어 핵심 요약</h1>
      <p class="hero-description">문서의 맥락을 읽고, 중요한 내용을 하나의 요약문으로 정리합니다.<br>뉴스와 보도자료를 읽는 마케팅·IT 기획자의 작업 공간입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    release = read_release()
    if release is None:
        st.info("모델 학습과 평가를 진행하고 있습니다. 최종 확인을 마치면 이 화면에서 요약할 수 있습니다.")
        st.caption("마케팅·고객·커머스, IT 제품·플랫폼·디지털 서비스 문단을 대상으로 준비 중입니다.")
        st.stop()


    @st.cache_resource
    def tokenizer_resource():
        return load_tokenizer()


    @st.cache_resource
    def model_resource(directory, digest):
        return load_model(ROOT / directory, digest)


    @st.cache_resource
    def selector_resource(digest):
        return load_selector()


    @st.cache_resource
    def rewriter_resource(base_digest, generator_digest):
        return load_rewriter()


    @st.cache_resource
    def factual_resources(release_digest):

        return load_factual_model(), SupportScorer()


    def clear_result():
        st.session_state.pop("result", None)
        for key in ("check_actor", "check_number", "check_status", "check_main"):
            st.session_state[key] = False


    examples_path = ROOT / "artifacts/demo_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8")) if examples_path.exists() else []
    example_by_name = {row["label"]: row for row in examples}


    def select_example():
        name = st.session_state["example"]
        st.session_state["source"] = example_by_name[name]["source"] if name in example_by_name else ""
        clear_result()


    def select_document():
        clear_result()
        st.session_state.pop("upload_error", None)
        uploaded = st.session_state.get("document")
        if uploaded is None:
            return
        try:
            st.session_state["source"] = read_document(uploaded.name, uploaded.getvalue())
            st.session_state["example"] = "직접 입력"
        except Exception as exc:
            st.session_state["source"] = ""
            st.session_state["upload_error"] = str(exc) if isinstance(exc, ValueError) else "문서의 본문을 읽지 못했습니다. 본문을 복사해 입력해 주세요."


    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="section-number">01 / INPUT</div>', unsafe_allow_html=True)
        st.subheader("원문 입력")
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        rewrite_manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
        modes = ["문서 맥락 요약 · 개발 중", "기존 문단 요약 · 비교용"] if rewrite_manifest else ["기존 문단 요약 · 비교용"]

        factual_release = json.loads(FACTUAL_RELEASE.read_text(encoding="utf-8")) if FACTUAL_RELEASE.exists() else {}
        if factual_release.get("adopted_in_app"):
            modes.insert(0, "관계·반복 개선 · 개발 중")
        mode = st.radio("요약 방식", modes, key="summary_mode", horizontal=True, on_change=clear_result)
        use_factual = mode.startswith("관계")
        use_rewrite = use_factual or mode.startswith("문서")
        st.file_uploader("문서 첨부", type=["pdf", "docx", "txt"], key="document", on_change=select_document,
                         help="본문 텍스트가 있는 PDF·Word·TXT 파일을 읽습니다. 스캔 이미지와 표의 배치는 지원하지 않습니다.")
        if st.session_state.get("upload_error"):
            st.warning(st.session_state["upload_error"])
        st.selectbox("예시 문단", ["직접 입력", *example_by_name], key="example", on_change=select_example)
        text = st.text_area("뉴스·보도자료 본문", key="source", height=330,
                            placeholder="문서를 첨부하거나 한 주제의 본문을 붙여 넣으세요.", on_change=clear_result)
        st.caption("문서 읽기 → 핵심 문장 선별 → 요약문 생성. 입력 원문은 서버에서 처리하며 자동 저장하지 않습니다." if use_rewrite else "비교용 기존 모델은 한 번에 한 문단을 요약합니다.")
        if use_rewrite:
            st.caption("개발 중인 방식입니다. 긴 문서의 핵심 누락과 문장 연결성은 검증을 이어가고 있습니다.")
        if st.button("요약문 생성하기", type="primary", use_container_width=True):
            clear_result()
            text = preprocess_news(text)
            with st.expander("요약에 사용할 정리된 본문"):
                st.write(text or "정리 후 남은 본문이 없습니다.")
            if len(text.strip()) < 30:
                st.warning("내용을 파악할 수 있도록 30자 이상의 문단을 입력해 주세요.")
            else:
                try:
                    tokenizer = tokenizer_resource()
                    count = len(input_ids(text, tokenizer))
                    limit = MAX_DOCUMENT_TOKENS if use_rewrite else release["max_source_tokens"]
                    if count > limit:
                        st.warning(f"현재 본문이 너무 깁니다({count}토큰). 한 주제의 본문을 나눠 입력해 주세요. 지원 범위는 {limit}토큰이며 원문을 자동으로 자르지 않습니다.")
                    else:
                        with st.spinner("문맥을 읽고 요약문을 작성하고 있습니다…"):
                            if use_factual:


                                model, scorer = factual_resources(sha256(FACTUAL_RELEASE))
                                selector = selector_resource(rewrite_manifest["selector_sha256"])
                                st.session_state["result"] = summarize_factual(text, model, tokenizer, selector, scorer)
                            elif use_rewrite:
                                generator = json.loads(GENERATOR_MANIFEST.read_text(encoding="utf-8")) if GENERATOR_MANIFEST.exists() else {}
                                model = rewriter_resource(release["weights_sha256"], generator.get("weights_sha256"))
                                selector = selector_resource(rewrite_manifest["selector_sha256"])
                                st.session_state["result"] = summarize_document(text, model, tokenizer, selector)
                            else:
                                model = model_resource(release["model_directory"], release["weights_sha256"])
                                st.session_state["result"] = generate(text, model, tokenizer,
                                    release["generation_settings"], release["max_source_tokens"])
                except (ValueError, AssertionError) as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("모델을 실행하지 못했습니다. 터미널 실행 상태와 모델 파일을 확인해 주세요.")
                    raise

    with right:
        st.markdown('<div class="section-number">02 / SUMMARY</div>', unsafe_allow_html=True)
        st.subheader("한눈에 읽는 요약문")
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        result = st.session_state.get("result")
        if result:
            summary_text = result["summary"] or "요약 문장이 생성되지 않았습니다. 다른 문단을 입력해 주세요."
            st.markdown('<div class="summary-paper"><div class="summary-label">생성 요약 초안</div>'
                        f'<div class="summary-copy">{escape(summary_text)}</div></div>', unsafe_allow_html=True)
            st.caption(f"생성 시간 {result['seconds']:.1f}초 · 입력 {result['source_tokens']}토큰")
            if result.get("pipeline") == "trained_rewriter_with_support_reranking":
                st.caption("원문의 주체와 내용을 대조해 고른 요약 초안입니다. 대조 기능도 오류를 놓칠 수 있습니다.")
            if result.get("requires_review"):
                st.warning("원문이 요약 내용을 충분히 뒷받침하는지 불확실합니다. 기업·제품과 기능의 연결을 확인해 주세요.")
            if result.get("repetition_cleanup"):
                st.caption("원문에 없는 연속 중복 단어를 정리했습니다.")
                with st.expander("중복 정리 전 생성문 확인"):
                    st.write(result["raw_generated_summary"])
            if result["numbers_not_in_source"]:
                st.warning("원문에서 찾지 못한 숫자 표현: " + ", ".join(result["numbers_not_in_source"]))
            if result["repeated_word_trigrams"] or result.get("adjacent_repeated_words"):
                st.warning("반복된 표현이 있습니다. 문장을 확인해 주세요.")
            if result.get("verbatim_source_span"):
                st.info("이번 요약은 원문 표현을 그대로 유지했습니다. 새로운 문장으로 충분히 재구성되었는지 검토해 주세요.")
            if result.get("numbers_not_in_evidence"):
                st.warning("선별된 문장에서 확인되지 않는 숫자가 있습니다. 원문과 대조해 주세요.")
            if result.get("selected_sentences"):
                with st.expander(f"요약에 사용한 핵심 문장 {len(result['selected_sentences'])}개 확인"):
                    for sentence in result["selected_sentences"]:
                        st.markdown(f'<div class="source-excerpt">{escape(sentence)}</div>', unsafe_allow_html=True)
                    st.caption("모델이 선택한 원문입니다. 선택된 문장만으로 충분한지, 빠진 핵심 내용은 없는지 확인하세요.")
            else:
                with st.expander("표현이 비슷한 원문 확인"):
                    for sentence in result["related_source_sentences"]:
                        st.markdown(f'<div class="source-excerpt">{escape(sentence)}</div>', unsafe_allow_html=True)
                    st.caption("생성 뒤 표현 유사도로 찾은 문장이며, 모델의 선별 결과는 아닙니다.")
            st.markdown("**사용 전 확인**")
            st.checkbox("기관·기업·인물과 발언 주체가 같습니다.", key="check_actor")
            st.checkbox("숫자·금액·날짜가 원문과 같습니다.", key="check_number")
            st.checkbox("계획·완료 상태와 원인·결과가 바뀌지 않았습니다.", key="check_status")
            st.checkbox("가장 중요한 사건이나 행동이 남아 있습니다.", key="check_main")
            st.download_button("요약과 원문 저장", data=f"[요약]\n{result['summary']}\n\n[원문]\n{result['source']}\n",
                               file_name="한국어_핵심요약.txt", mime="text/plain; charset=utf-8")
        else:
            st.markdown('''<div class="empty-paper"><div class="summary-label">요약 준비</div>
            <div class="empty-title">중요한 내용을 모아<br>하나의 요약문으로.</div>
            <p>문서를 읽어 중요한 문장을 고른 뒤<br>생성 모델이 요약문을 작성합니다.<br>사용한 핵심 문장은 펼쳐서 확인할 수 있습니다.</p>
            <p>주체, 수치, 상태, 핵심 내용을 대조한 뒤<br>필요한 결과를 저장하세요.</p></div>''', unsafe_allow_html=True)

    st.divider()
    st.caption("학습 데이터: 과학기술정보통신부·한국지능정보사회진흥원 AI Hub ‘요약문 및 레포트 생성 데이터’(582). 개인 연구·학습용 시제품입니다.")
    with st.expander("모델과 사용 범위"):
        st.write("한국어 KoBART의 문맥 표현과 학습한 문장 선별기를 사용하고, 선별 내용을 KoBART 생성 모델에 함께 전달합니다. 입력한 문서는 별도의 기록 파일에 자동 저장하지 않습니다.")
        st.write("새 방식은 한 주제의 문서를 최대 4,096토큰까지 나누어 읽습니다. 긴 문서의 품질, 일반 대화, 법률·의료 판단은 평가 범위에 포함하지 않았습니다. 생성 길이는 최대 112토큰이며 핵심 누락이나 주체 변경이 남을 수 있습니다.")
        st.caption("원문과 요약의 주체·수치·상태·핵심 내용을 대조한 뒤 사용하세요.")

if __name__ == "__main__":
    main()
