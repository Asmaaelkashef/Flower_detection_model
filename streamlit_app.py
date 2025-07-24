import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input  # ✅ هنا الإضافة
from PIL import Image

# تحميل الموديل
model = load_model("flower_model.keras")

# أسماء الفئات
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# العنوان
st.title("🌼 Flower Detector")
st.markdown("Upload a flower image and I’ll tell you what kind it is!")

# رفع الصورة
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # فتح الصورة
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    try:
        # معالجة الصورة
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))  # ✅ preprocessing الصحيح

        # التنبؤ
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        # النتيجة
        st.markdown(f"### 🌸 I think this is a **:violet[{predicted_class}]**!")

    except Exception as e:
        st.error("❌ Error predicting the flower.")
        st.code(str(e))

# التوقيع اللطيف
st.markdown("---")
st.markdown("Made with ❤️ by **Asmaa Elkashef**")
