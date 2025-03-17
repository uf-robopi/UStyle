# **UStyle: Waterbody Style Transfer of Underwater Scenes by Depth-Guided Feature  Synthesis**
<div align="center">
  <img src="images/Figure2_6.png" alt="UStyle intro" width="700">
</div>

---

## Models and Files
1. A pre-trained ResNet50 encoder and a blockwisely trained decoder which reconstructs intermediate feature maps and the input image. Blockwise training enables the decoder to progressively learn to reproduce lower-level features.
    - The model is defined in `model.py` and its checkpoints are saved in the `checkpoints` directory.
2. A fusion model that integrates content and style features via a depth-aware whitening and coloring transform (DA-WCT) blending. This model enhances stylization by fusing features from multiple scales while incorporating depth information.
    - The fusion models are implemented in `fusion_1to1.py` and `fusion_all.py`.
    - Guided filtering for post-processing is implemented in `utils/photo_gif.py` (adapted from the [PhotoWCT code](https://github.com/NVIDIA/FastPhotoStyle)).

## Training
`train.py` is the training code for our ResNet-based blockwise model. Usage instructions and hyperparameters are provided within the file. Fine-tuning can be performed using `finetune.py`.

---

## Requirements 
- Python 3.10
- PyTorch (tested with version torch 1.8.1+cu111)
- torchvision 0.9.1+cu111, numpy, opencv-python, Pillow, and other standard libraries

---
  
## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/siddique-ab/UStyle.git
   cd UStyle/
2. Setup the environment:
   ```bash
   conda env create -f environment.yml
   conda activate UStyle
3. Train and Finetune UStyle:
   ```bash
   python3 train.py
   python3 finetune.py
4. Inference using UStyle:
   ```bash
   python3 inference_1to1.py
   python3 inference_all.py

---

<h2>Bibliography</h2>
<div id="bibtex">
  <pre>
@inproceedings{siddique2024aquafuse,
    author={Siddique, Md Abu Bakr and Liu, Junliang and Singh, Piyush and Islam, Md Jahidul},
    title={UStyle: Waterbody Style Transfer of Underwater Scenes by Depth-Guided Feature Synthesis},
    booktitle={arXiv preprint},
    year={2025}
}
  </pre>
</div>


---

## Acknowledgements
- https://github.com/NVIDIA/FastPhotoStyle
- https://github.com/chiutaiyin/PhotoWCT2
