import json
import os
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from threading import Thread
from typing import Dict
from transformers import AutoTokenizer
from optimum.rbln import BatchTextIteratorStreamer, RBLNLlamaForCausalLM
import logging


class ConversationManager:
    def __init__(self):
        self.user_conversations = {}
        self.user_summary_history = {}
        self.user_input_counts = {}
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.case_data = {}
        self.compiled_model = None
        self.tokenizer = None
        self.batch_size = 3
        self.max_seq_len = 8192

    def get_or_compile_model(
        self,
        model_save_dir_prefix: str,
        model_id: str,
        batch_size: int,
        max_seq_len: int,
        tp: int = 4,
    ):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        model_save_dir = f"{model_save_dir_prefix}.model_id={model_id}.batch_size={batch_size}.max_seq_len={max_seq_len}"
        logger.info(f"Model save directory: {model_save_dir}")

        if not os.path.exists(model_save_dir):  # 모델이 없으면 컴파일
            logger.info("Model not found. Compiling new model...")
            try:
                compiled_model = RBLNLlamaForCausalLM.from_pretrained(
                    model_id=model_id,
                    export=True,
                    rbln_batch_size=batch_size,
                    rbln_max_seq_len=max_seq_len,
                    rbln_tensor_parallel_size=tp,
                )
                logger.info("Model compilation successful")

                compiled_model.save_pretrained(model_save_dir)
                logger.info(f"Model saved to {model_save_dir}")

            except Exception as e:
                logger.error(f"Error during model compilation: {str(e)}")
                raise

        logger.info("Loading pre-compiled model...")
        try:
            compiled_model = RBLNLlamaForCausalLM.from_pretrained(
                model_id=model_save_dir,
                export=False,
            )
            logger.info("Model loaded successfully")
            return compiled_model

        except Exception as e:
            logger.error(f"Error loading compiled model: {str(e)}")
            raise

    def initialize_user_conversation(self, user_id):
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = [
                {
                    "role": "system",
                    "content": "처음부터 끝까지 일관되게 한국어로 대답해줘. 너는 은행 업무 전문가야.",
                }
            ]
            self.user_summary_history[user_id] = [
                {
                    "role": "system",
                    "content": "너는 대화 내용을 요약해주는 시스템이야. 사용자가 말한 내용을 간단히 요약해줘.",
                }
            ]

        # 시스템 메시지를 제외한 대화 기록이 8개(4쌍)를 초과하면 오래된 대화 제거
        if len(self.user_conversations[user_id]) > 9:  # 시스템 메시지(1) + 대화쌍(4쌍=8개) = 9
            system_message = self.user_conversations[user_id][0]
            self.user_conversations[user_id] = [system_message] + self.user_conversations[user_id][-8:]

    def search_similar_cases(self, user_input, top_k=3):
        user_embedding = self.embedding_model.encode([user_input])
        distances, indices = self.index.search(np.array(user_embedding), top_k)
        return [self.case_data[idx] for idx in indices[0]]

    def generate_response(self, user_id, message):
        """
        사용자 입력에 대한 응답을 생성하는 함수

        user_id: 사용자 ID
        message: 사용자 입력 새로운 메시지
        """

        # 시스템 메시지 설정
        system_message = {
            "role": "system",
            "content": (
                "모든 응답은 한국어로만 해야 돼. 영어를 포함한 다른 언어는 사용하지 말고, 처음부터 끝까지 일관되게 한국어로 대답해줘. "
                "너는 은행 업무 전문가로서, 송금 대상과의 관계 및 송금 목적을 파악하고 그에 대한 구체적인 정보를 수집해야 해. "
                "이전 대화 내역을 참고해서, 사용자의 송금 목적에 대해 가능한 선택지를 제공하며, 자연스럽게 대화를 이어나가도록 해줘."
            ),
        }

        # 최근 5개의 대화 맥락과 새로운 메시지를 포함하여 프롬프트를 생성
        conversation_context = self.user_conversations[user_id][
            -5:
        ] + [  # 최근 5개 대화만 포함
            {"role": "user", "content": message}
        ]
        conversation_with_system = [system_message] + conversation_context

        # 토크나이저를 사용하여 대화 내역을 토큰화
        input_text = self.tokenizer.apply_chat_template(
            conversation_with_system, add_generation_prompt=True, tokenize=False
        )
        input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=True)

        # 입력 길이 확인 및 조정
        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        logging.debug(f"input_ids shape: {input_ids.shape}")
        logging.debug(f"attention_mask shape: {attention_mask.shape}")

        # 길이가 max_seq_len을 초과하면 잘라내기
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            attention_mask = attention_mask[:, : self.max_seq_len]

        # 빈 입력이 있는지 확인 후 처리
        if input_ids.size(1) == 0 or attention_mask.size(1) == 0:
            raise RuntimeError(
                "Empty input_ids or attention_mask, check input length and context size."
            )

        # 입력을 배치 크기에 맞게 확장하여 모델에 전달
        input_ids_expanded = input_tokens["input_ids"].expand(self.batch_size, -1)
        attention_mask_expanded = input_tokens["attention_mask"].expand(
            self.batch_size, -1
        )

        # 디버깅: input_ids와 attention_mask의 크기 확인
        logging.debug(f"input_ids_expanded shape: {input_ids_expanded.shape}")
        logging.debug(f"attention_mask_expanded shape: {attention_mask_expanded.shape}")

        # `step`, `max_seq_len`, `prefill_chunk_size` 확인을 위한 디버깅 출력
        try:
            # `max_seq_len` 확인
            logging.debug(f"Model sequence length (max_seq_len): {self.max_seq_len}")

            # `prefill_chunk_size` 확인
            if hasattr(self.compiled_model, "prefill_chunk_size"):
                logging.debug(
                    f"Prefill chunk size: {self.compiled_model.prefill_chunk_size}"
                )
            else:
                logging.debug(
                    "Compiled model does not have prefill_chunk_size attribute"
                )

        except Exception as e:
            logging.error(f"Error accessing model configuration: {e}")

        # 모델에 입력하여 응답 생성
        response_output = self.compiled_model.generate(
            input_ids=input_ids_expanded,
            attention_mask=attention_mask_expanded,
            # max_length=max_seq_len,
            max_new_tokens=150,  # 생성할 토큰 수를 150으로 제한
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )

        # 로그 생성된 응답을 디버깅
        logging.debug(f"Generated response output: {response_output}")

        # 생성된 응답을 디코딩하여 단일 응답으로 반환
        response_text = self.tokenizer.decode(
            response_output[0], skip_special_tokens=True
        )

        # 디버깅: 디코딩된 응답 텍스트 확인
        logging.debug(f"Decoded response text: {response_text}")

        return response_text

    def generate_summary_and_search(self, user_id):
        try:
            print("\n=== 5회 대화 요약 시작 ===")
            
            # 전체 대화 텍스트 추출
            user_text = "\n".join(
                [
                    entry["content"] 
                    for entry in self.user_conversations[user_id] 
                    if entry["role"] == "user"
                ]
            )
            
            # 유사 사례 검색
            similar_cases = self.search_similar_cases(user_text)
            
            # 보이스피싱 위험도 평가
            risk_level, alert, keywords = self.assess_phishing_risk(user_text, similar_cases)
            
            # 요약 생성 시 위험도 정보 포함
            summary_prompt = self.user_summary_history[user_id] + [
                {
                    "role": "user", 
                    "content": f"지금까지의 대화를 요약하고, 보이스피싱 위험도({risk_level})와 주의사항도 포함해서 알려줘."
                }
            ]
            
            # 요약 프롬프트 토큰화
            summary_text = self.tokenizer.apply_chat_template(
                summary_prompt, add_generation_prompt=True, tokenize=False
            )
            summary_input = self.tokenizer(summary_text, return_tensors="pt", padding=True)
            
            # 입력을 배치 크기에 맞게 확장
            input_ids_expanded = summary_input["input_ids"].expand(self.batch_size, -1)
            attention_mask_expanded = summary_input["attention_mask"].expand(
                self.batch_size, -1
            )
            
            # 요약 생성
            summary_output = self.compiled_model.generate(
                input_ids=input_ids_expanded,
                attention_mask=attention_mask_expanded,
                max_new_tokens=300,  # 생성할 토큰 수를 제한
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
            )
            
            # 요약 내용 디코딩
            summary = self.tokenizer.decode(summary_output[0], skip_special_tokens=True)
            print("\n[요약 결과]:")
            print(summary)
            
            # 요약 내용 로그 기록
            logging.info(f"Summary for {user_id}: {summary}")
            
            # 요약 내용을 대화에 추가
            self.user_conversations[user_id].append(
                {"role": "assistant", "content": summary}
            )
            
            # 유사 사례를 대화에 추가
            print("\n=== 유사 사례 ===")
            for case in similar_cases:
                case_message = f"사례: {case['scenario']}, 대응: {case['response']}"
                print(case_message)  # 콘솔 출력
                self.user_conversations[user_id].append(
                    {"role": "assistant", "content": case_message}
                )
            
            print("=== 5회 대화 요약 완료 ===\n")
            
        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            logging.error(f"Error in generate_summary_and_search: {str(e)}")
            raise

    def assess_phishing_risk(self, conversation_text, similar_cases):
        try:
            print("\n=== 보이스피싱 위험도 평가 ===")
            
            # 위험 키워드 정의
            high_risk_keywords = [
                "급함", "긴급", "즉시", "빨리", "최대한도", 
                "다른 명의", "타인", "계좌", "신분증",
                "검사", "검찰", "수사관", "경찰", "금감원",
                "돈세탁", "범죄", "연루", "압수", "동결"
            ]
            
            # 위험도 점수 계산
            risk_score = 0
            detected_keywords = []
            
            for keyword in high_risk_keywords:
                if keyword in conversation_text:
                    risk_score += 1
                    detected_keywords.append(keyword)
            
            # 유사 사례 기반 위험도 가중치
            if similar_cases:
                risk_score += len(similar_cases) * 0.5
            
            # 위험도 레벨 결정
            if risk_score >= 5:
                risk_level = "매우 높음"
                alert = "⚠️ 보이스피싱 위험이 매우 높습니다! 즉시 통화를 종료하고 금융감독원(1332)에 신고하세요."
            elif risk_score >= 3:
                risk_level = "높음"
                alert = "⚠️ 보이스피싱이 의심됩니다. 신중한 판단이 필요합니다."
            elif risk_score >= 1:
                risk_level = "주의"
                alert = "💡 안전한 금융거래를 위해 주의가 필요합니다."
            else:
                risk_level = "낮음"
                alert = "✅ 현재까지는 특별한 위험이 감지되지 않았습니다."
            
            # 위험도 평가 결과 출력
            print(f"\n위험도 레벨: {risk_level}")
            if detected_keywords:
                print(f"감지된 위험 키워드: {', '.join(detected_keywords)}")
            print(f"주의사항: {alert}")
            print("\n유사 보이스피싱 사례:")
            for case in similar_cases:
                print(f"- 사례: {case['scenario']}")
                print(f"- 대응방법: {case['response']}\n")
            
            print("=== 위험도 평가 종료 ===\n")
            
            return risk_level, alert, detected_keywords
            
        except Exception as e:
            print(f"위험도 평가 중 오류 발생: {str(e)}")
            raise

    def process_user_input(self, user_id, message):
        try:
            logging.info(f"Initializing conversation for user: {user_id}")
            self.initialize_user_conversation(user_id)

            # 사용자 입력을 기록
            logging.info(f"Recording user input for user: {user_id}")
            self.user_conversations[user_id].append({"role": "user", "content": message})

            # 응답 생성 전 알림
            print("\n응답 생성 중...\n")

            # 응답 생성
            logging.info(f"Generating response for user: {user_id}")
            response_text = self.generate_response(user_id, message)

            # 응답 출력
            print("="*50)
            print("[AI 응답]:")
            print(response_text)
            print("="*50)

            # 대화 기록에 응답 추가
            self.user_conversations[user_id].append(
                {"role": "assistant", "content": response_text}
            )
            
            # 대화 횟수 증가
            self.user_input_counts[user_id] = self.user_input_counts.get(user_id, 0) + 1
            print(f"\n현재 대화 횟수: {self.user_input_counts[user_id]}/5")

            # 5회 대화마다 요약 및 위험도 평가
            if self.user_input_counts[user_id] == 5:
                logging.info(f"Generating summary and searching similar cases for user: {user_id}")
                self.generate_summary_and_search(user_id)
                self.user_input_counts[user_id] = 0  # 카운터 초기화

            return response_text

        except Exception as e:
            logging.error(f"Error in process_user_input: {str(e)}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"  # 사용자 친화적 에러 메시지


if __name__ == "__main__":
    try:
        manager = ConversationManager()

        # 보이스피싱 JSON 데이터 로드
        with open("voice_phishing_cases.json", "r", encoding="utf-8") as file:
            voice_phishing_cases = json.load(file)

        # 시나리오 텍스트 벡터화
        manager.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        case_embeddings = [
            manager.embedding_model.encode(case["scenario"])
            for case in voice_phishing_cases
        ]
        
        # FAISS 인덱스에 추가하기 전에 입력 데이터 확인
        if not case_embeddings or not isinstance(case_embeddings[0], np.ndarray):
            raise ValueError("Invalid embeddings provided for FAISS index.")

        # FAISS 인덱스 생성
        dimension = case_embeddings[0].shape[0]
        manager.index = faiss.IndexFlatL2(dimension)
        manager.index.add(np.array(case_embeddings))

        # 사례를 인덱스와 함께 저장
        manager.case_data = {i: case for i, case in enumerate(voice_phishing_cases)}

        # 모델 설정
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model_save_dir = "rbln-Llama-3-8B-Instruct"
        batch_size = 3
        max_seq_len = 8192
        tp = 4  # 텐서 병렬 처리

        # 사용자 대화 이력과 배치 입력 관리
        manager.user_conversations = {f"user_{i+1}": [] for i in range(batch_size)}
        manager.user_summary_history = {f"user_{i+1}": [] for i in range(batch_size)}
        manager.user_input_counts = {f"user_{i+1}": 0 for i in range(batch_size)}

        # 모델과 토크나이저 초기화
        manager.compiled_model = manager.get_or_compile_model(
            "prefix", model_id, batch_size, max_seq_len
        )
        manager.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        manager.tokenizer.pad_token = manager.tokenizer.eos_token

        # 스트리밍을 위한 Streamer 설정
        streamer = BatchTextIteratorStreamer(
            tokenizer=manager.tokenizer,
            batch_size=batch_size,
            skip_special_tokens=False,
            skip_prompt=False,
        )
        print("\n=== 대화형 모드 시작 ===")
        while True:
            try:
                user_id = input("\n사용자 ID를 입력하세요 (user_1, user_2, user_3): ").strip()
                
                if user_id not in ["user_1", "user_2", "user_3"]:
                    print("WARNING: 올른 사용자 ID를 입력하세요.")
                    continue
                    
                user_input = input(f"{user_id} 입력: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n대화를 종료합니다.")
                    break
                    
                # 사용자 입력 처리 및 응답 생성
                manager.process_user_input(user_id, user_input)
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"오류가 발생했습니다: {str(e)}")
                continue
    finally:
        # 필요한 정리 작업 수행
        pass
