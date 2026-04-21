# 📄 Paper Translation

> **PDF 학술 논문을 한국어로 번역하는 오픈소스 도구**
> 이미지와 표는 원본 그대로 보존하고, 텍스트만 GPT-4o로 번역하여 PDF로 출력합니다.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-supported-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 목차

- [소개](#소개)
- [주요 기능](#주요-기능)
- [동작 원리](#동작-원리)
- [프로젝트 구조](#프로젝트-구조)
- [시작하기](#시작하기)
  - [사전 요구사항](#사전-요구사항)
  - [환경 변수 설정](#환경-변수-설정)
  - [Linux — 직접 실행](#linux--직접-실행)
  - [Windows — Docker 실행](#windows--docker-실행)
- [사용 방법](#사용-방법)
- [기여하기](#기여하기)
- [라이선스](#라이선스)

---

## 소개

학술 논문을 읽을 때 언어 장벽은 큰 걸림돌입니다. **Paper Translation**은 PDF 논문을
[docling](https://github.com/DS4SD/docling)으로 파싱하고,
OpenAI GPT-4o API를 통해 텍스트만 선택적으로 번역합니다.

기존 번역 도구들은 표나 이미지까지 깨뜨리거나, 레이아웃을 손상시키는 경우가 많습니다.
이 프로젝트는 **표·이미지는 절대 건드리지 않고**, 순수 텍스트 블록만 번역하여
원문의 구조를 최대한 유지한 PDF를 생성하는 것을 목표로 합니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **선택적 번역** | 텍스트(`<p>`, `<h1>~<h2>`)만 번역, 표·이미지는 원본 유지 |
| **고품질 번역** | GPT-4o 기반, 전문 학술 번역 프롬프트 적용 |
| **다중 입력** | PDF 파일 업로드 또는 URL(arXiv 등) 직접 입력 |
| **이중 출력** | 번역된 PDF와 HTML 동시 다운로드 |
| **진행 상황 표시** | 실시간 번역 진행 바 및 현재 블록 미리보기 |
| **한국어 폰트** | Noto Serif CJK KR 적용으로 깔끔한 한글 렌더링 |
| **크로스 플랫폼** | Linux 직접 실행 / Windows Docker 실행 |

---

## 동작 원리

```
PDF 입력 (파일 또는 URL)
        │
        ▼
  [docling 파싱]
   구조화된 문서 요소로 분리
        │
        ├──► TableItem   ──► HTML <table> 그대로 삽입 (번역 없음)
        │
        ├──► PictureItem ──► base64 <img> 그대로 삽입 (번역 없음)
        │
        └──► TextItem    ──► GPT-4o 번역 ──► <p> / <h1> / <h2>
                                                    │
                                                    ▼
                                           [WeasyPrint]
                                        HTML → PDF 변환 출력
```

번역 시 다음 규칙을 프롬프트로 강제합니다:

- 기술 용어·고유명사·약어는 원문 유지 (또는 괄호 병기)
- 원문의 어조·문단 구조 보존
- 요약·설명·주석 추가 금지
- 한국어 번역문만 출력

---

## 프로젝트 구조

```
Paper_Translation/
├── app.py                  # Streamlit UI 진입점
├── core/
│   ├── __init__.py
│   ├── styles.py           # PDF·HTML 출력용 CSS (Noto CJK 폰트 포함)
│   ├── translator.py       # OpenAI API 호출 및 번역 로직
│   ├── document.py         # docling 문서 파싱, HTML 빌드
│   └── pipeline.py         # 전체 변환 파이프라인 (파싱→번역→PDF)
├── Dockerfile              # WeasyPrint 의존성 포함 Linux 이미지
├── docker-compose.yml      # .env 주입, output 볼륨 마운트
├── .dockerignore
├── requirements.txt
└── .env                    # 직접 생성 필요 (아래 참고)
```

**`core/` 모듈 역할:**

- **`styles.py`** — 출력 PDF·HTML에 적용되는 CSS를 상수로 관리합니다.
- **`translator.py`** — OpenAI client를 받아 단일 텍스트 블록을 번역합니다. UI 의존성이 전혀 없어 단독 테스트가 가능합니다.
- **`document.py`** — docling 문서를 순회하며 요소 타입별로 처리합니다. 번역 진행 상황은 `on_progress` 콜백으로 외부에 전달하여 UI와 분리됩니다.
- **`pipeline.py`** — 파싱·번역·PDF 생성을 하나로 묶는 퍼사드(facade)입니다. `app.py`는 이 함수만 호출합니다.

---

## 시작하기

### 사전 요구사항

| 항목 | Linux | Windows |
|---|---|---|
| Python | 3.11 이상 | 불필요 |
| pip 패키지 | `requirements.txt` | 불필요 |
| 시스템 라이브러리 | Cairo, Pango (아래 참고) | 불필요 |
| Docker | 선택 사항 | **필수** |
| OpenAI API 키 | 필수 | 필수 |

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=sk-...
```

> API 키가 없으면 앱 사이드바에서 직접 입력할 수도 있습니다.

---

### Linux — 직접 실행

**1. 시스템 의존성 설치** (WeasyPrint가 Cairo·Pango를 필요로 합니다)

```bash
# Ubuntu / Debian
sudo apt install -y \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 shared-mime-info \
    fonts-noto-cjk tesseract-ocr tesseract-ocr-kor
```

**2. Python 패키지 설치**

```bash
pip install -r requirements.txt
```

**3. 앱 실행**

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

---

### Windows — Docker 실행

WeasyPrint는 Windows 네이티브 환경에서 Cairo·Pango 설치가 복잡하므로,
Docker 컨테이너 내부의 Linux 환경에서 실행합니다.

**사전 설치:** [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)

**1. 이미지 빌드 및 컨테이너 시작**

```bash
docker compose up --build
```

최초 실행 시 이미지 빌드에 수 분이 소요될 수 있습니다.
이후 실행부터는 캐시를 사용하여 빠르게 시작됩니다.

**2. 접속**

브라우저에서 `http://localhost:8501` 로 접속합니다.

**3. 컨테이너 종료**

```bash
docker compose down
```

---

## 사용 방법

1. 사이드바에서 **OpenAI API 키**를 입력합니다. (`.env`에 설정했다면 자동 입력)
2. **입력 방식**을 선택합니다.
   - `파일 업로드` — PDF 파일을 드래그하거나 선택합니다.
   - `URL 입력` — arXiv 등 공개 PDF URL을 붙여넣습니다. (예: `https://arxiv.org/pdf/2408.09869`)
3. **번역 시작** 버튼을 클릭합니다.
4. 번역이 완료되면 **PDF 다운로드** 또는 **HTML 다운로드** 버튼이 나타납니다.

> **참고:** 논문 길이에 따라 수 분이 소요될 수 있습니다. 텍스트 블록 수만큼 OpenAI API 호출이 발생하므로 API 사용량에 유의하세요.

---

## 기여하기

기여는 언제나 환영합니다. 버그 리포트, 기능 제안, Pull Request 모두 가능합니다.

```bash
# 1. 저장소 포크 후 클론
git clone https://github.com/parkjunsu3321/Paper_Translation.git
cd Paper_Translation

# 2. 브랜치 생성
git checkout -b feat/your-feature

# 3. 변경 후 커밋
git commit -m "feat: 설명"

# 4. Push 및 Pull Request 생성
git push origin feat/your-feature
```

---

## 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.
