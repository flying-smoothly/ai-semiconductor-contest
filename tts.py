from google.cloud import texttospeech

# Text-to-Speech 클라이언트 생성
client = texttospeech.TextToSpeechClient()

# 한국어 음성 설정
voice_kor = texttospeech.VoiceSelectionParams(
    language_code="ko-KR",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)

# 텍스트 설정
input_text = texttospeech.SynthesisInput(text="안녕하세요! 입금하려면, 먼저 원하는 금액과 계좌 정보를 알려주세요. 그 후에 확인 절차를 진행하겠습니다. 계좌 번호가 123-456-7890이 맞나요? 입금할 금액은 50만원입니다. 입금을 진행할까요?")

# 오디오 설정 (AudioConfig를 직접 사용)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# 음성 합성 요청
response = client.synthesize_speech(
    input=input_text, voice=voice_kor, audio_config=audio_config
)

# 음성 파일을 지정한 경로에 저장 (예: C 드라이브의 tts_output 폴더)
save_path ="C:/Users/STORY/Desktop/ai 반도체 필요한 코드/output_korean.mp3"

with open(save_path, "wb") as out:
    out.write(response.audio_content)
    print(f'음성 파일이 "{save_path}"로 저장되었습니다.')
    
    #텍스트 동적으로 받아와야. 음성 파일 지정한 경로에 저장하는 부분 수정 필요