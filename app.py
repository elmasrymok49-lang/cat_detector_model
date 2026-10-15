import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 1. إعدادات الصفحة وتصميم الـ UI
st.set_page_config(
    page_title="AI Cat & Dog Detector", page_icon="🤖", layout="centered"
)

# تصميم عصري 3D Glassmorphism
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        border: none;
        width: 100%;
        padding: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #818cf8;'>⚡ النظام الذكي المحلي"
    " للرؤية الحاسوبية</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>ارفع صورة أو التقط صورة"
    " مباشرة لتحليلها بالذكاء الاصطناعي المحلي.</p>",
    unsafe_allow_html=True,
)


# 2. هيكل النموذج العصبي
class CatDogCNN(nn.Module):

  def __init__(self):
    super(CatDogCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(32 * 8 * 8, 64)
    self.fc2 = nn.Linear(64, 2)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 32 * 8 * 8)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x


@st.cache_resource
def load_model():
  model = CatDogCNN()
  model.load_state_dict(
      torch.load("cat_detector_model.pth", map_location=torch.device("cpu"))
  )
  model.eval()
  return model


try:
  model = load_model()
except Exception as e:
  st.error(
      "⚠️ برجاء التأكد من وجود ملف نموذج التدريب 'cat_detector_model.pth' بجانب"
      " ملف app.py"
  )

# خيار رفع الصورة أو التصوير
uploaded_file = st.file_uploader(
    "📁 اختر صورة من جهازك (قطة أو كلب)", type=["jpg", "jpeg", "png"]
)
camera_file = st.camera_input("📸 أو التقط صورة بالكاميرا مباشرة")

image = None
if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
elif camera_file is not None:
  image = Image.open(camera_file).convert("RGB")

if image is not None:
  st.image(image, caption="الصورة التي تم اختيارها", use_column_width=True)

  if st.button("🔍 ابدأ تحليل الذكاء الاصطناعي"):
    with st.spinner("جاري التفكير وتحليل الأنماط البصرية..."):
      # معالجة الصورة
      transform = transforms.Compose(
          [
              transforms.Resize((32, 32)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ]
      )
      image_tensor = transform(image).unsqueeze(0)

      with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, dim=1)

      confidence = max_prob.item()

      # نظام الذكاء الواعي
      if confidence < 0.65:
        st.info("💡 لم أتعلم هذا إلى الآن، ولكن قريباً!")
      else:
        class_names = ["قطة 🐱", "كلب 🐶"]
        result = class_names[predicted.item()]
        st.success(
            f"🧠 **النتيجة:** {result}  |  **نسبة الثقة:**"
            f" {int(confidence * 100)}%"
        )
