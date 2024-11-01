import requests
import json


class ClovaSpeechClient:
    # Clova Speech invoke URL (앱 등록 시 발급받은 Invoke URL)
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/9089/b304adc33719d6cddb96d6c4f1aa46628d927547b49b70c981c7fea953e0600a'
    # Clova Speech secret key (앱 등록 시 발급받은 Secret Key)
    secret = 'cf453525306a4cf4b71763e123acb090'

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None, sed=None):
        # 요청에 필요한 메타데이터 설정
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret  # API Key를 헤더에 포함
        }

        # 파일과 메타데이터를 함께 전송할 파일 설정
        files = {
            'media': open(file, 'rb'),  # 업로드할 파일
            'params': (None, json.dumps(request_body, ensure_ascii=False), 'application/json')  # JSON 파라미터
        }

        # POST 요청 전송
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)

        # 응답 결과 반환
        return response


if __name__ == '__main__':
    # ClovaSpeechClient 객체 생성
    client = ClovaSpeechClient()

    # 파일 업로드 요청
    result = client.req_upload(file=r"C:\Users\STORY\Desktop\ai_code\CSR\시끄러운 상황 + 작은 목소리.m4a", completion='sync')

    # JSON 응답에서 'text'만 추출하여 출력
    result_json = result.json()
    if 'text' in result_json:
        print(result_json['text'])  # 'text' 필드만 출력
    else:
        print("No 'text' field found in response.")


