# **UStyle: Waterbody Style Transfer of Underwater Scenes by Depth-Guided Feature  Synthesis**
<div align="center">
  <img src="./model.png" alt="UStyle intro" width="700">
</div>

---
## Pointers
- Preprint: 
- UF7D Dataset: [DropBox Link](https://www.dropbox.com/scl/fo/ab7eim7wh5ezlwfh43v6l/AHCChi7xj9stdV48AT1Ati0?rlkey=tnp2lbay9nfhe24c4qlw9gpjn&st=k4eh0s69&dl=0)
- UStyle checkpoints and qualitative comparison files are also in the same folder

## Models and Files
1. The model is defined in `model.py` 
2. Download and save the checkpoints in the `checkpoints/` directory.
3. Our fusion model integrates content and style features via a depth-aware whitening and coloring transform (DA-WCT) blending. This model enhances waterbody stylization by fusing features from multiple scales while incorporating depth information
    - The fusion models are implemented in `fusion_1to1.py` and `fusion_all.py`.
    - Guided filtering for post-processing is implemented in `utils/photo_gif.py` (adapted from the [PhotoWCT code](https://github.com/NVIDIA/FastPhotoStyle)).
4. `train.py` is the training code for our ResNet-based blockwise model. 
5. Fine-tuning can be performed using `finetune.py`.

---

## Requirements 
- Python 3.10
- PyTorch (tested with version torch 1.8.1+cu111)
- torchvision 0.9.1+cu111, numpy, opencv-python, Pillow, and other standard libraries

---
  
## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/uf-robopi/UStyle.git
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
@article{siddique2025ustyle,
    author={Siddique, Md Abu Bakr and Liu, Junliang and Singh, Piyush and Islam, Md Jahidul},
    title={UStyle: Waterbody Style Transfer of Underwater Scenes by Depth-Guided Feature Synthesis},
    journal={ArXiv Preprint},
    year={2025}
}
  </pre>
</div>


---

## Acknowledgements
- https://github.com/NVIDIA/FastPhotoStyle
- https://github.com/chiutaiyin/PhotoWCT2
