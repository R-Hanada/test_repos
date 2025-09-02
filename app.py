import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title('YOLOv8 物体検知Webアプリ')

@st.cache_resource
def load_model():
    model = YOLO('runs/train/exp8/weights/best.pt')
    return model

model = load_model()

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # 推論ボタンを押すまでアップロード画像を表示
    infer = st.button('推論する')
    if not infer:
        st.image(image, caption='アップロード画像', use_column_width=True)

    if infer:
        # PIL画像をnumpy配列に変換
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 推論
        results = model(img_bgr)

        # バウンディングボックス付き画像取得
        result_img = results[0].plot()  # numpy array (BGR)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # 推論結果のみ表示
        st.image(result_img_rgb, caption='推論結果', use_column_width=True)
