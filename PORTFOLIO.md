# 프로젝트 소개: AI 반도체 기반 보이스피싱 탐지 금융 챗봇

## 프로젝트 개요

보이스피싱 시도를 실시간으로 탐지하고 금융 관련 상담을 제공하는 RAG 기반 대화형 AI 챗봇을 구현한 프로젝트입니다. 190팀이 참가한 AI 반도체 기술인재 선발대회에서 최종 5팀에 선발되었으며, 2024 인공지능반도체 미래기술 컨퍼런스에 참가해 성과를 발표했습니다. NAVER Clova Speech로 음성을 텍스트로 변환한 뒤, AI 반도체(RBLN NPU)에 최적화된 LLaMA 모델로 위험도를 분석하는 파이프라인을 구축했습니다.

## 기술 스택 및 아키텍처

`Python`을 기반으로 Rebellions의 NPU 추론 SDK인 `optimum-rbln`의 `RBLNLlamaForCausalLM`을 통해 LLaMA 모델을 AI 반도체 하드웨어에 맞게 컴파일·배포합니다. 유사 보이스피싱 사례 검색에는 `FAISS` 벡터 인덱스와 `SentenceTransformer`(`all-MiniLM-L6-v2`) 임베딩 모델을 조합한 RAG 파이프라인을 구성했습니다. 음성 입력은 `NAVER Clova Speech API`의 `req_upload()` 메서드를 통해 처리하며, 환경 변수로 API 키를 주입받는 방식으로 보안과 이식성을 확보했습니다.

## 핵심 구현 로직

`ConversationManager` 클래스가 시스템의 중심으로, `get_or_compile_model()` 메서드에서 지정 경로에 컴파일된 모델이 없을 경우에만 `RBLNLlamaForCausalLM.from_pretrained(export=True)`로 NPU 컴파일을 수행하고 이후에는 캐시된 모델을 바로 로드하는 방식으로 초기화 비용을 줄였습니다. 대화 맥락 관리는 슬라이딩 윈도우 방식을 채택해 시스템 메시지를 포함해 최대 9개(4쌍 대화 + 시스템 메시지)만 유지하고 초과 시 오래된 기록을 제거하며, 별도의 요약 전용 대화 히스토리(`user_summary_history`)를 병행 관리해 장기 맥락 손실을 완화합니다. `search_similar_cases()`는 사용자 입력을 임베딩한 뒤 FAISS `index.search()`로 상위 3개의 유사 사례를 검색해 RAG 컨텍스트로 LLM 프롬프트에 주입하는 구조로, 모델의 환각을 줄이고 실제 사례 기반 탐지 신뢰도를 높이는 판단이 반영된 것으로 보입니다.
