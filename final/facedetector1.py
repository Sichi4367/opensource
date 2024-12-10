import os
import cv2
import numpy as np
import tensorflow as tf

# 모델 경로 설정
model_path = "./facedata/model.savedmodel"
labels_path = "./facedata/labels.txt"

# 모델 로드
if not os.path.exists(model_path):
    raise FileNotFoundError(f"SavedModel 파일을 찾을 수 없습니다: {model_path}")

model = tf.saved_model.load(model_path)

# 라벨 로드
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {labels_path}")

with open(labels_path, "r", encoding="utf-8") as f:
    labels = f.read().strip().split("\n")

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 가져올 수 없습니다.")
        break

    # 모델 입력 데이터 준비
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)  # 모델 입력 크기로 리사이즈
    normalized_frame = resized_frame.astype(np.float32) / 255.0  # 정규화
    input_data = np.asarray(resized_frame, dtype=np.float32)  # 배치 차원 추가

    input_data = (input_data / 127.5) - 1

    input_data = np.expand_dims(input_data, axis=0)
    
    # 모델 예측
    predictions = model(input_data)
    smoothed_predictions = predictions
    alpha = 0.9  # 이전 값에 대한 가중치 (0~1)
    smoothed_predictions = alpha * smoothed_predictions + (1 - alpha) * predictions
    max_idx = np.argmax(smoothed_predictions[0])  # 가장 높은 확률의 인덱스
    label = labels[max_idx]
    confidence = smoothed_predictions[0][max_idx] * 100

    # 결과 표시
    result_text = f"{label}: {confidence:.2f}%"

    # 화면에 결과 표시
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 하단 흰색 바 생성
   

    # 통합된 화면 출력
    cv2.imshow('Animal Prediction', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()