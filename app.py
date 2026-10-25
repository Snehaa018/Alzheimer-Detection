
import re, pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="Alzheimer's Disease Detection",
                   page_icon="🧠", layout="centered")


# ── MRI model (must match the training class) ───────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + self.shortcut(x))


class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── Loading ─────────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open("classes.pkl", "rb") as f:
        classes = pickle.load(f)
    cnn = AlzheimerCNN(num_classes=len(classes))
    cnn.load_state_dict(torch.load("alzheimer_cnn.pth", map_location="cpu"))
    cnn.eval()
    with open("clinical_model.pkl", "rb") as f:
        clin = pickle.load(f)
    return cnn, classes, clin

cnn, classes, clin = load_all()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Helpers ─────────────────────────────────────────────────
def norm(name):
    n = re.sub(r"[^a-z]", "", str(name).lower())
    return n.replace("dementia", "demented")


def align(p, src_classes, tgt_classes):
    idx = {norm(c): i for i, c in enumerate(src_classes)}
    out = np.array([p[idx[norm(c)]] if norm(c) in idx else 0.0 for c in tgt_classes])
    s = out.sum()
    return out / s if s > 0 else out


def clinical_probs(values):
    row = pd.DataFrame([values], columns=clin["features"])
    row["Gender"] = row["Gender"].map(clin["gender_map"])
    model = clin["model"]
    proba = model.predict_proba(row)[0]
    full = np.zeros(len(clin["clin_classes"]))
    full[model.classes_] = proba
    return align(full, clin["clin_classes"], classes)


COLORS = {"nondemented": "green", "verymilddemented": "orange",
          "milddemented": "orange", "moderatedemented": "red"}


# ── UI ──────────────────────────────────────────────────────
st.title("🧠 Alzheimer's Disease Detection")
st.markdown("Upload a brain MRI and, optionally, enter clinical values. "
            "The MRI model and the clinical model each give probabilities, "
            "and they are combined by weighted averaging (decision-level fusion).")
st.divider()

uploaded = st.file_uploader("MRI image", type=["jpg", "jpeg", "png"])
img = None
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

use_clin = st.checkbox("Add clinical data", value=False)
values = None
w_img = 0.5
if use_clin:
    r = clin["ranges"]
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", list(clin["gender_map"].keys()))
        age = st.number_input("Age at entry", min_value=float(r["ageAtEntry"][0]),
                              max_value=float(r["ageAtEntry"][1]),
                              value=float(r["ageAtEntry"][2]), step=1.0)
        cdr = st.selectbox("CDR score", clin["cdr_values"])
    with c2:
        mmse = st.number_input("MMSE score", min_value=float(r["mmse"][0]),
                               max_value=float(r["mmse"][1]),
                               value=float(r["mmse"][2]), step=1.0)
        memory = st.number_input("Memory score", min_value=float(r["memory"][0]),
                                 max_value=float(r["memory"][1]),
                                 value=float(r["memory"][2]), step=0.1)
    w_img = st.slider("Weight of MRI model in the final result", 0.0, 1.0, 0.5, 0.05)
    values = {"Gender": gender, "mmse": mmse, "ageAtEntry": age,
              "cdr": cdr, "memory": memory}

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    if img is None:
        st.error("Please upload an MRI image first!")
    else:
        with st.spinner("Analysing..."):
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                p_img = torch.softmax(cnn(x), dim=1)[0].numpy()

            if values is not None:
                p_clin = clinical_probs(values)
                final = w_img * p_img + (1 - w_img) * p_clin
            else:
                p_clin = None
                final = p_img

        pred = int(np.argmax(final))
        pred_class = classes[pred]
        color = COLORS.get(norm(pred_class), "blue")

        st.subheader("Prediction Result")
        st.markdown(f"<h2 style='color:{color};'>{pred_class}</h2>",
                    unsafe_allow_html=True)
        st.metric("Confidence", f"{final[pred] * 100:.1f}%")

        st.subheader("Class Probabilities")
        table = {"MRI model": p_img}
        if p_clin is not None:
            table["Clinical model"] = p_clin
            table["Fused"] = final
        st.bar_chart(pd.DataFrame(table, index=classes))

        if p_clin is not None:
            st.caption("CDR and MMSE are clinical scores that closely define the "
                       "diagnosis, so the clinical model is expected to be very "
                       "confident. Adjust the fusion weight above if needed.")

        st.warning("⚠️ For educational purposes only. "
                   "Always consult a qualified physician for diagnosis.")
