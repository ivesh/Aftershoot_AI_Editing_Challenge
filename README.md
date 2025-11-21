
---

# 📸 Aftershoot AI Editing Challenge

### **Predicting Accurate & Consistent White Balance Adjustments (Temperature & Tint)**

At **Aftershoot**, we build **AI-powered tools for professional photographers** to automate the most time-consuming parts of their workflow—culling, editing, and retouching—so they can spend more time behind the camera and less time in front of a computer.

Our models use **Machine Learning, Deep Learning, and Computer Vision** to understand images deeply:
sharpness, emotion, lighting, skin tones, subject focus, and more.
These insights then flow downstream into advanced editing models capable of **matching a photographer’s personal style**.

This challenge reflects a **real production problem** faced in our AI Editing pipeline.

---

# 🧩 Background: Non-Destructive Editing at Aftershoot

Instead of directly modifying RAW files, Aftershoot performs **non-destructive editing**:

1. All edits are stored as **sliders** inside a **SQLite database**.
2. During export, Aftershoot merges the **RAW image + slider edits** to generate a final JPEG.

To learn a user’s editing style, photographers provide:

* A catalog of ~5000 edited images
* Corresponding RAW files
* SQLite-edit slider values

These sliders (floats) become the **training targets** for ML models.

---

# 🎯 Problem Overview

You must build a model that predicts **White Balance Temperature** and **Tint** values, given:

* The image
* As-shot WB values
* Camera metadata
* Environmental features
* Additional numeric/categorical attributes

The challenge stems from:

### ✔ Non-linear Temperature sensitivity

A 500K shift at **2000–2500K** visibly impacts the image more than the same shift at **5000–5500K**.

### ✔ Linear Tint scale

Tint shifts uniformly from **−150 to +150**.

### ✔ As-shot white balance varies by camera brand

This serves as an anchor point for the model.

### ✔ Consistency requirement

Small changes (extra light source, subject color, zoom-in/out) modify pixel distributions and cause inconsistent predictions—while the resulting images should have **similar edits**.

Your model must be:

1. **Accurate**
2. **Consistent across visually similar images**
3. **Fast**
4. **Lightweight**

You have **full freedom**: image-only models, metadata-only, hybrid fusion networks, regression/classification, sequential models—anything.

---

# 📁 Dataset Structure

The provided dataset contains two folders: `Train/` and `Validation/`.

```
dataset/
│
├── Train/
│   ├── images/              # 2,539 TIFF images (256x256)
│   └── sliders.csv          # Input features + labels
│
└── Validation/
    ├── images/              # 493 TIFF images (256x256)
    └── sliders_inputs.csv   # Input features only (no labels)
```

---

# 📌 Data Columns Overview

### **ID**

* `id_global` — image identifier (maps to TIFF filename)

### **As-Shot White Balance**

* `currTemp`
* `currTint`

### **EXIF Metadata**

* `aperture`
* `flashFired`
* `focalLength`
* `isoSpeedRating`
* `shutterSpeed`

### **Camera Information**

* `camera_model`
* `camera_group`

### **Extra Properties**

* `intensity`
* `ev`

### **Target Variables**

* `Temperature`
* `Tint`

---

# 🏆 Task

Build a machine learning model that predicts:

* `Temperature`
* `Tint`

given:

* the corresponding TIFF image
* all input features from sliders CSV

You may use:

* CNNs
* Vision+Metadata fusion models
* Tabular ML models
* Regression / classification / hybrid
* Any loss function of your choice

---

# 🧪 Evaluation Metric

We evaluate using **Mean Absolute Error (MAE)**:

```
mae_temperature = 1 / (1 + MAE(actual_Temp, predicted_Temp))
mae_tint        = 1 / (1 + MAE(actual_Tint, predicted_Tint))
```

Final score is computed similarly for both predictions.

Your submission **must contain predictions** for both:

* `Temperature`
* `Tint`

---

# 📤 Submission Format

Submit a **CSV file** with:

```
id_global,Temperature,Tint
IMG_00001,5120,7
IMG_00002,4780,12
...
```

* The `id_global` must match filenames in `Validation/images/`
* Both target columns are **mandatory**

---

# 💡 Suggested Approaches (Optional Inspiration)

* CNN + MLP fusion
* Vision Transformer with tabular side-channel
* Metadata-only baseline
* Image-only baseline
* Mixture-of-experts
* Normalized temperature scaling
* Consistency-driven loss terms
* Camera-aware models
* Contrastive learning between “similar-looking” images

---

# 🚀 Running the Code (example template)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --config configs/default.yaml

# Generate validation predictions
python infer.py --weights checkpoints/best_model.pth
```

---


