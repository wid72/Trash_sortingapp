try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "opencv-python-headless"])
    import cv2
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# 모델 로드
model = YOLO("yolov8n.pt")  # 처음엔 가장 작은 모델 사용

# Streamlit UI
st.title("🗑️ 스마트 쓰레기 분류기")
st.write("이미지를 업로드하면, YOLOv8로 분류된 쓰레기 종류를 알려드립니다.")

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="업로드된 이미지", use_column_width=True)

    with st.spinner("분석 중..."):
        results = model(img_array, conf=0.4)
        result_image = img_array.copy()

        labels = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                labels.append(f"{label} ({conf:.2f})")

                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.image(result_image, caption="분석 결과", use_column_width=True)
        st.success(f"감지된 객체: {', '.join(labels)}")
