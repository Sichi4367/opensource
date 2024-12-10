import cv2
import numpy as np
import tensorflow.keras.models as tf_model

# 모델과 라벨 파일 경로 설정
model_path = "/mnt/data/keras_model.h5"
labels_path = "/mnt/data/labels.txt"

# 모델 및 라벨 로드
model = tf_model.load_model(model_path)

# 라벨 로드
with open(labels_path, "r", encoding="utf-8") as f:
    labels = f.read().strip().split("\n")

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 영상을 가져올 수 없습니다.")
        break

    # 모델 입력 크기에 맞게 프레임 전처리
    resized_frame = cv2.resize(frame, (224, 224))  # 모델에 맞는 크기
    normalized_frame = resized_frame / 255.0      # 정규화
    input_data = np.expand_dims(normalized_frame, axis=0)

    # 모델 예측
    predictions = model.predict(input_data)
    max_idx = np.argmax(predictions[0])
    max_label = labels[max_idx]
    confidence = predictions[0][max_idx] * 100

    # 결과 텍스트 준비
    result_text = f"{max_label}: {confidence:.2f}%"

    # 프레임 표시
    cv2.imshow("Webcam", frame)

    # 하단 하얀창 생성
    white_bar = np.ones((100, 500, 3), dtype=np.uint8) * 255
    cv2.putText(white_bar, result_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    combined_frame = cv2.vconcat([frame, white_bar])

    # 통합 창 출력
    cv2.imshow("Webcam with Prediction", combined_frame)

    # 종료 조건 (q 키 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
