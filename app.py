import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import sys
import os

# YOLOv5 코드 경로를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# 디바이스 설정
device = select_device('cpu')

# 모델 로드
model = DetectMultiBackend('yolov5s.pt', device=device, dnn=False)
model.eval()

# Streamlit UI
st.title("🗑️ YOLOv5 스마트 쓰레기 분류기")
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    img0 = np.array(image)
    img = letterbox(img0, new_shape=(640, 640))[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(img0, caption="분석 결과", use_column_width=True)
