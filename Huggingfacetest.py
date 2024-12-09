
from transformers import pipeline


# Zero-shot 이미지 분류 파이프라인 생성
def classify_animal_huggingface(image_path):
    """
    Hugging Face 모델을 사용하여 이미지에 나타난 동물을 분류합니다.
    """
    # Hugging Face의 zero-shot-image-classification 모델 로드
    classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

    # 이미지를 열어 모델에 전달
    with open(image_path, "rb") as image:
        categories = ["dog", "cat", "bird", "lion", "tiger", "elephant", "horse", "fish", "snake"]
        result = classifier(image, candidate_labels=categories)
    
    # 결과 출력
    if result:
        return result[0]['label']  # 가장 높은 확률의 라벨 반환
    else:
        return "분류 실패"

# 테스트 이미지 경로
image_path = "example.jpg"  # 테스트할 이미지 경로
result = classify_animal_huggingface(image_path)
print(f"예측된 동물: {result}")
