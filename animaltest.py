import openai
from PIL import Image
import requests
from io import BytesIO

# OpenAI API 키 설정
openai.api_key = "YOUR_OPENAI_API_KEY"

def classify_animal(image_path):
    """
    이미지 파일 경로를 받아 동물 이름을 반환합니다.
    """
    # 이미지를 불러오기
    with open(image_path, "rb") as image_file:
        # OpenAI API에 이미지 업로드
        response = openai.Image.create(
            file=image_file,
            purpose="fine_tune"
        )

    # 이미지에 대한 설명 생성
    if response:
        image_id = response['id']
        print("이미지 분석 중...")
        # 이미지 기반 설명 요청
        description = openai.Completion.create(
            model="gpt-4",
            prompt=f"This is an image of {image_id}. What animal is in the picture?",
            temperature=0.5,
            max_tokens=50
        )
        # 결과 반환
        return description['choices'][0]['text'].strip()
    else:
        return "이미지 분석 실패."

# 테스트 실행
image_path = "example.jpg"  # 분석할 이미지 경로
result = classify_animal(image_path)
print(f"동물 종류: {result}")
