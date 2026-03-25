# Vision Pipeline: ResNet-50 From Scratch for Pascal VOC 2012

> A clean, notebook-first computer vision project for training a **ResNet-50 from scratch** on **Pascal VOC 2012**, with Docker support, training progress feedback, and a best-model visualization block.

## ✨ Highlights

- 🧠 Custom **ResNet-50** implementation built from scratch with Bottleneck blocks
- 🚫 **No pretrained weights** and **no `torchvision.models`**
- 🖼️ Trains on **Pascal VOC 2012** using local images and XML annotations
- 🎯 Multi-label classification with **20 VOC classes**
- 📈 Progress bars, step-wise metric updates, loss tracking, and validation metrics
- 🧪 Random-image visualization using the **best saved checkpoint**
- 🐳 Docker workflow included for reproducible runs
- 💾 Checkpoints are saved automatically during training

## 🗂️ Project Layout

```text
.
├── Vision_model.ipynb
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
├── VOCdevkit/                  # downloaded dataset lives here
│   └── VOC2012/
│       ├── Annotations/
│       ├── ImageSets/
│       └── JPEGImages/
├── resnet50_voc_best.pth       # auto-generated after training
└── resnet50_voc_final.pth      # auto-generated after training
```

## 📦 Dataset

This project is built around **Pascal VOC 2012**.

- Official VOC 2012 page: <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/index.html>
- Official train/val archive: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>

### What the notebook expects

After extraction, the dataset should look like this:

```text
VOCdevkit/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt
    │       └── val.txt
    └── JPEGImages/
```

### Download and extract

If you prefer manual download:

1. Open the official VOC 2012 page.
2. Download the **training/validation data** archive.
3. Extract it into the project root so that `VOCdevkit/VOC2012` exists beside the notebook.

If you want a terminal flow:

```powershell
# From the project root
tar -xf VOCtrainval_11-May-2012.tar
```

### Dataset summary used by the notebook

- Train images: **5,717**
- Validation images: **5,823**
- Total train/val images seen by the notebook: **11,540**
- Classes: **20**

### Important note

Pascal VOC images can contain **multiple labels per image**, so this project uses **multi-label classification** instead of single-label classification.

## 🧰 Local Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch and Torchvision

Install the version of **PyTorch** and **Torchvision** that matches your machine:

- PyTorch install guide: <https://pytorch.org/get-started/locally/>

### 3. Install project dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Jupyter

```powershell
jupyter notebook
```

Then open:

```text
Vision_model.ipynb
```

## 🐳 Docker Setup

The included Dockerfile uses a **PyTorch CUDA runtime image** and starts a Jupyter Notebook server inside the container.

### Build the image

```powershell
docker build -t vision-pipeline .
```

### Run with GPU

```powershell
docker run --rm -it --gpus all -p 8888:8888 -v "${PWD}:/app" vision-pipeline
```

### Run without GPU

```powershell
docker run --rm -it -p 8888:8888 -v "${PWD}:/app" vision-pipeline
```

### Docker tips

- Mounting the project folder with `-v "${PWD}:/app"` is the easiest workflow.
- Your local notebook edits, dataset, and auto-saved checkpoints stay on the host machine.
- The Jupyter URL with the access token will appear in the container logs.
- Make sure `VOCdevkit/VOC2012` exists inside the mounted project folder before training.

## 🧠 Model Architecture

This project uses a **hand-written ResNet-50** built from scratch.

### High-level flow

```text
Input image (3 x 224 x 224)
    ->
7x7 Conv, 64 filters, stride 2
    ->
BatchNorm + ReLU
    ->
3x3 MaxPool
    ->
Layer 1: 3 Bottleneck blocks
    ->
Layer 2: 4 Bottleneck blocks
    ->
Layer 3: 6 Bottleneck blocks
    ->
Layer 4: 3 Bottleneck blocks
    ->
Adaptive Average Pool
    ->
Fully Connected Layer
    ->
20 logits
    ->
Sigmoid
    ->
Multi-label predictions
```

### Bottleneck block idea

```text
input
  |---------------- skip connection ----------------|
  v                                                 |
1x1 conv -> 3x3 conv -> 1x1 conv -------------------+
                        |
                        v
                       add
                        |
                       ReLU
```

### Pseudo architecture notes

- The network follows the canonical ResNet-50 block layout: **[3, 4, 6, 3]**
- The final linear layer maps features to **20 VOC classes**
- Since this is **classification**, the model predicts **which classes are present**
- The model does **not** predict bounding boxes

## ⚙️ Training Configuration

Current notebook hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Image size | 224 |
| Batch size | 16 |
| Epochs | 8 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss |
| Prediction threshold | 0.5 |
| Metric update frequency | every 100 steps |
| Best checkpoint path | `resnet50_voc_best.pth` |
| Final checkpoint path | `resnet50_voc_final.pth` |

### Training behavior

- Uses **CUDA automatically** if available
- Shows a **progress bar** for training and validation
- Prints loss and accuracy updates every **100 steps**
- Tracks train and validation loss/accuracy per epoch
- Saves the best and final checkpoints automatically

## 📊 Current Reported Result

- Current reported validation accuracy: **92.8%**

### Metric note

This notebook reports a **multi-label threshold accuracy** using `threshold = 0.5`.
That is helpful for project tracking, but it is **not the same as Pascal VOC mAP**.

## 💾 Model Saving

The trained model files are **not tracked in Git**.

That is intentional:

- `*.pth` is ignored by Git
- checkpoints are generated locally during training
- the notebook saves:
  - `resnet50_voc_best.pth`
  - `resnet50_voc_final.pth`

So even if the repository clone does not contain any `.pth` file, training will create them for you automatically.

## 🎨 Visualization Extras

The notebook includes two visualization helpers:

### 1. Batch preview

Shows a sample batch from the dataloader so you can quickly confirm:

- image loading works
- transforms are correct
- labels look reasonable

### 2. Best-model random sample viewer

At the end of the notebook, there is an extra block that:

- loads the **best saved model**
- picks a **random validation image**
- draws the **ground-truth VOC rectangles** from the XML annotation
- writes the object names on the image
- prints the semantic predictions from the best checkpoint

### Important visualization note

The rectangles come from the **dataset annotations**, not from the classifier itself.
The classifier contributes the **semantic labels and confidences** shown for the image.

## 🚀 How To Use

### Train

1. Download and extract Pascal VOC 2012.
2. Make sure `VOCdevkit/VOC2012` is in the project root.
3. Open `Vision_model.ipynb`.
4. Run the notebook from top to bottom.

### After training

You will get:

- the best checkpoint
- the final checkpoint
- training and validation metrics
- plots for loss history
- best-model visualization on a random validation image

## 📝 Notes

- The notebook is designed to be **readable and beginner-friendly**
- The project favors **accuracy first** and **speed second**
- The architecture is implemented from scratch for learning clarity
- The final visualization block is especially useful for sanity-checking both annotations and semantic predictions

## 🙌 Credits

- Dataset: Pascal VOC 2012 by the VOC organizers
- Framework: PyTorch
- Notebook workflow: Jupyter

---

If you train the notebook on your own setup, the checkpoints will be created automatically and kept locally, even though they are not stored in Git.
