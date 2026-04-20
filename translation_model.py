"""
MIT 라이선스 기준 번역 예시 (facebook/m2m100).

- NLLB는 CC-BY-NC 등으로, MIT-only 배포와는 맞지 않을 수 있음.
- m2m100은 **문장 단위 다대다 MT**라 논문 문장이 **직역·번역투**로 보이기 쉽습니다.
  자연스러운 논문체는 **문맥(앞뒤 문단)**, **용어집**, **후편집**, 또는 **LLM 기반 번역·교정**을
  파이프라인에 붙이는 식으로 보완하는 경우가 많습니다.

HF MIT 필터: https://huggingface.co/models?license=license:mit
"""

import os

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 418M: 가볍게 테스트 / 1.2B: 같은 라이선스(MIT 태그)에서 품질 상승, VRAM·다운로드 증가
model_id = os.environ.get("M2M100_MODEL", "facebook/m2m100_418M")

tokenizer = M2M100Tokenizer.from_pretrained(model_id)
model = M2M100ForConditionalGeneration.from_pretrained(model_id)

# 번역 방향 (예: ko -> en 으로 바꿔서 비교 가능)
SRC_LANG = "en"
TGT_LANG = "ko"

# Attention 논문 첫 문장 스타일 (직역체가 잘 드러나는 예시)
text = (
    "Recurrent neural networks, convolutional neural networks"
)

tokenizer.src_lang = SRC_LANG
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)

# seq2seq에서 max_length는 보통 **디코더가 만들 출력** 상한(인코더 길이와 별개)
max_decode_length = min(320, max(96, int(inputs["input_ids"].shape[1] * 2) + 32))

outputs = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.get_lang_id(TGT_LANG),
    max_length=max_decode_length,
    num_beams=8,
    early_stopping=True,
    length_penalty=1.05,
    no_repeat_ngram_size=4,
    repetition_penalty=1.02,
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)