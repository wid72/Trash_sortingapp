import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

st.title("🗑️ YOLOv5 기반 스마트 쓰레기 분류기")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    img = np.array(image)
    results = model(img)

    # 결과 시각화
    result_img = np.squeeze(results.render())  # YOLO의 결과 이미지
    st.image(result_img, caption="분석 결과", use_column_width=True)

    labels = results.pandas().xyxy[0]['name'].value_counts()
    st.write("🔍 감지된 객체:")
    st.write(labels)
