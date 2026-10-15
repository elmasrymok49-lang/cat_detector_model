import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# إعدادات الواجهة الاحترافية
st.set_page_config(
    page_title="Pro AI Cat & Dog Classifier", page_icon="⚡", layout="centered"
)

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white; border-radius: 12px; font-weight: bold; border: none; width: 100%; padding: 10px;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4); }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #818cf8;'>⚡ النظام الذكي الخارق للرؤية"
    " الحاسوبية</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>مدعوم بنموذج MobileNetV2"
    " الاحترافي بدقة تحليل عالية جداً.</p>",
    unsafe_allow_html=True,
)


# تحميل النموذج الذكي الخارق
@st.cache_resource
def load_model():
  model = models.mobilenet_v2(weights=None)
  model.classifier[1] = nn.Linear(model.last_channel, 2)
  model.load_state_dict(
      torch.load("cat_detector_model.pth", map_location=torch.device("cpu"))
  )
  model.eval()
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      "⚠️ برجاء التأكد من رفع ملف النموذج الجديد 'cat_detector_model.pth' بجانب"
      " app.py"
  )

uploaded_file = st.file_uploader(
    "📁 اختر صورة واضحة (قطة أو كلب)", type=["jpg", "jpeg", "png"]
)
camera_file = st.camera_input("📸 أو التقط صورة بالكاميرا مباشرة")

image = None
if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
elif camera_file is not None:
  image = Image.open(camera_file).convert("RGB")

if image is not None:
  st.image(image, caption="الصورة المختارة للتحليل", use_column_width=True)

  if st.button("🔍 تحليل ذكي فائق الدقة"):
    with st.spinner("جاري تفكيك الأنماط البصرية بدقة مذهلة..."):
      # تحويلات بجودة 128x128 المتوافقة مع النموذج الخارق
      transform = transforms.Compose([
          transforms.Resize((128, 128)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      ])
      image_tensor = transform(image).unsqueeze(0)

      with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, dim=1)

      confidence = max_prob.item()

      # نظام الثقة والذكاء الواعي
      if confidence < 0.70:
        st.info("💡 لم أتعلم هذا إلى الآن، ولكن قريباً!")
      else:
        class_names = ["قطة 🐱", "كلب 🐶"]
        result = class_names[predicted.item()]
        st.success(
            f"🎯 **النتيجة الدقيقة:** {result}  |  **نسبة الثقة المرتفعة:**"
            f" {int(confidence * 100)}%"
        )
