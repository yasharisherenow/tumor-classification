# Tumor Classification using Deep Learning (PyTorch + ResNet50)

This project implements a deep learning pipeline using a fine-tuned ResNet-50 model to classify brain tumors from MRI images. The model supports multi-label classification across several tumor types and is optimized for GPU usage with mixed-precision training.

---

## Tumor Types Supported

- Glioma
- Meningioma
- Pituitary
- No Tumor

The classification task is multi-label, allowing a single image to be associated with one or more tumor types.

---

## Model Details

- Backbone: `resnet50` (pretrained on ImageNet)
- Final Layer: Custom linear output for multi-label classification
- Loss Function: `BCEWithLogitsLoss` with `pos_weight` for handling class imbalance
- Optimizer: Adam
- Scheduler: `ReduceLROnPlateau`
- Mixed-Precision: Enabled with `torch.cuda.amp`

---

## Test Results

| Class        | Precision | Recall | F1 Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 1.00      | 1.00   | 1.00     | 145     |
| Meningioma   | 0.99      | 0.99   | 0.99     | 123     |
| No Tumor     | 0.99      | 1.00   | 1.00     | 115     |
| Pituitary    | 0.98      | 0.99   | 0.99     | 154     |
| Macro Avg    | 0.99      | 1.00   | 0.99     | 537     |
| Micro Avg    | 0.99      | 1.00   | 0.99     | 537     |

Final test F1 score: **0.9937**

Confusion matrices and misclassified image visualizations are saved to the `outputs/` directory.

---

## Project Structure

```

tumor-classification/
├── tumor\_model.py         # Full training and inference pipeline
├── app.py                 # (optional) Streamlit app for live prediction
├── outputs/               # Model checkpoints and visualizations
├── data/                  # train/, valid/, test/ each with \_classes.csv
├── requirements.txt
└── README.md

````

---

## Training the Model

```bash
python tumor_model.py \
  --data_path ./data \
  --out ./outputs \
  --visualize \
  --amp \
  --epochs 30
````

Each split (`train/`, `valid/`, `test/`) should contain a `_classes.csv` file with image filenames and binary labels for each class.

Example `_classes.csv`:

```csv
filename,glioma,meningioma,pituitary,no_tumor
img1.jpg,1,0,0,0
img2.jpg,0,1,1,0
```

---

## Inference on a Single Image

```bash
python tumor_model.py \
  --inference \
  --model_path outputs/model_inference.pth \
  --image_path ./sample_inputs/test_image.jpg
```

The script prints prediction probabilities and saves a visualization to `./inference_results/`.

---

## Streamlit App (Optional)

To launch the interactive web app:

```bash
streamlit run app.py
```

The app allows users to upload images and see predicted tumor classes with confidence scores.

---

## Model Artifact

* File: `outputs/model_inference.pth`
* Size: \~90 MB
* Includes:

  * `model_state_dict`
  * `class_names`
  * `img_size` for inference

Consider using Git Large File Storage (LFS) for reliable version control of model weights.

---

## Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies:

* `torch`, `torchvision`
* `pandas`, `numpy`
* `scikit-learn`, `matplotlib`, `seaborn`
* `streamlit` (for app)

---

## Author

**Yasharisherenow**
GitHub: [tumor-classification](https://github.com/yasharisherenow/tumor-classification)
Developed using PyTorch for medical image analysis and deep learning exploration.

---


## License

This project is intended for research and educational use only. Not validated for clinical or diagnostic purposes. Use responsibly.


