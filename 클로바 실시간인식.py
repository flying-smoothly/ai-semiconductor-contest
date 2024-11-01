import grpc
import json
import nest_pb2
import nest_pb2_grpc

AUDIO_PATH = "C:\\Users\\STORY\\Desktop\\ai 반도체\\청파동2가.pcm"   #인식할 오디오 파일이 위치한 경로를 입력해 주십시오. (16kHz, 1channel, 16 bits per sample의 PCM (헤더가 없는 raw wave) 형식)
CLIENT_SECRET = "864d12b979e347538ef482e91995a474"

def generate_requests(audio_path):
    # 초기 설정 요청: 음성 인식 설정
    yield nest_pb2.NestRequest(
        type=nest_pb2.RequestType.CONFIG,
        config=nest_pb2.NestConfig(
            config=json.dumps({"transcription": {"language": "ko"}})
        )
    )

    # 오디오 파일을 열고 32,000 바이트씩 읽음
    with open(audio_path, "rb") as audio_file:
        while True:
            chunk = audio_file.read(32000)  # 오디오 파일의 청크를 읽음
            if not chunk:
                break  # 데이터가 더 이상 없으면 루프 종료
            print(f"Sending chunk of size: {len(chunk)} bytes")
            yield nest_pb2.NestRequest(
                type=nest_pb2.RequestType.DATA,
                data=nest_pb2.NestData(
                    chunk=chunk,
                    extra_contents=json.dumps({"seqId": 0, "epFlag": False})
                )
            )

def extract_text_from_responses(responses):
    texts = []

    for response in responses:
        try:
            # response가 이미 JSON 형식임을 감안하여 바로 접근
            response_data = json.loads(response.contents)
            # transcription 필드가 존재하는지 확인
            if 'transcription' in response_data and 'text' in response_data['transcription']:
                texts.append(response_data['transcription']['text'])
            else:
                print("No transcription field in response.")
        except (json.JSONDecodeError, KeyError):
            # JSON 파싱 오류나 KeyError가 발생할 경우 무시
            continue

    return ' '.join(texts)

def main():
    # Clova Speech 서버에 대한 보안 gRPC 채널을 설정
    channel = grpc.secure_channel(
        "clovaspeech-gw.ncloud.com:50051",
        grpc.ssl_channel_credentials()
    )
    stub = nest_pb2_grpc.NestServiceStub(channel)  # NestService에 대한 stub 생성
    metadata = (("authorization", f"Bearer {CLIENT_SECRET}"),)  # 인증 토큰과 함께 메타데이터 설정
    responses = stub.recognize(generate_requests(AUDIO_PATH), metadata=metadata)  # 생성된 요청으로 인식(recognize) 메서드 호출
    
    # 전체 텍스트를 저장할 리스트
    full_transcription = []
    
    try:
        # 서버로부터 받은 응답을 처리
        responses = stub.recognize(generate_requests(AUDIO_PATH), metadata=metadata)
        
        for response in responses:
            response_data = json.loads(response.contents)  # 응답을 JSON으로 파싱
            # transcription 필드가 존재하는지 확인
            if 'transcription' in response_data and 'text' in response_data['transcription']:
                print(f"Received response text: {response_data['transcription']['text']}")
                # 텍스트를 리스트에 추가
                full_transcription.append(response_data['transcription']['text'])
            else:
                print("No transcription field in this response.")

        # 리스트에 저장된 텍스트를 모두 합쳐서 출력
        extracted_text = ' '.join(full_transcription)
        print("Extracted text: ", extracted_text)

    except grpc.RpcError as e:
        # gRPC 오류 처리
        print(f"Error: {e.details()}")
        
    finally:
        channel.close()  # 작업 완료 후 채널 닫기

        

if __name__ == "__main__":
    main()