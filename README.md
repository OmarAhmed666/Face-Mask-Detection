# Adv.-AI — Face Mask Detection

A CNN that classifies whether a person is **wearing a mask** or **not**, built with TensorFlow/Keras and OpenCV.

- **Task:** Binary classification (`with_mask` / `without_mask`)
- **Input:** 224×224 RGB images
- **Test accuracy:** ~99%

## Repository Structure

```
Adv.-AI/
├── notebook.ipynb        # Data prep, training, evaluation
├── Interface code.py     # Real-time webcam inference
├── Model.h5               # Trained model weights
└── README.md
```

## Dataset

5,988 images across two folders (`with_mask`, `without_mask`), split 80/20 for train/test. Images are resized to 224×224 and normalized; training data is augmented (rotation, zoom, shift, flip).

## Model

A custom CNN with 4 convolutional blocks (Conv2D → BatchNorm → MaxPool, filters 32→64→128→128) followed by a dense classifier head (Dense(128) → Dropout(0.5) → Dense(2, softmax)).

Trained with Adam (`lr=1e-3`), binary cross-entropy loss, batch size 32, up to 20 epochs with early stopping.

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| with_mask | 1.00 | 0.98 | 0.99 |
| without_mask | 0.98 | 1.00 | 0.99 |

**Overall accuracy: 99%**

## Real-Time Inference

`Interface code.py` uses OpenCV's Haar Cascade to detect faces in a webcam feed, feeds each face through the trained model, and draws a green (mask) or red (no mask) box with the confidence %. Press `Q` to quit.

**Note:** the script defaults to loading `mask_detector_cnn.h5` — update `MODEL_PATH` to `Model.h5` to match this repo before running.

## Setup

```bash
pip install tensorflow==2.13.0 opencv-python numpy matplotlib scikit-learn imutils
```

Run training: `jupyter notebook notebook.ipynb`
Run detection: `python "Interface code.py"`

## Notes

- Built in Google Colab (T4 GPU) with the dataset on Google Drive — update paths to run locally.
- `Model.h5` (~37 MB) is tracked via Git/Git LFS.

## Author

Omar Abouellil — coursework project (Eyouth Advanced AI Track).
