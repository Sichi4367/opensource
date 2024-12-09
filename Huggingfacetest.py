
from transformers import pipeline


# Zero-shot �̹��� �з� ���������� ����
def classify_animal_huggingface(image_path):
    """
    Hugging Face ���� ����Ͽ� �̹����� ��Ÿ�� ������ �з��մϴ�.
    """
    # Hugging Face�� zero-shot-image-classification �� �ε�
    classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

    # �̹����� ���� �𵨿� ����
    with open(image_path, "rb") as image:
        categories = ["dog", "cat", "bird", "lion", "tiger", "elephant", "horse", "fish", "snake"]
        result = classifier(image, candidate_labels=categories)
    
    # ��� ���
    if result:
        return result[0]['label']  # ���� ���� Ȯ���� �� ��ȯ
    else:
        return "�з� ����"

# �׽�Ʈ �̹��� ���
image_path = "example.jpg"  # �׽�Ʈ�� �̹��� ���
result = classify_animal_huggingface(image_path)
print(f"������ ����: {result}")
