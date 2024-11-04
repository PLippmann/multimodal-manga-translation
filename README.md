# Multimodal Manga Translation
Repo for paper "Context-Informed Machine Translation of Manga using Multimodal Large Language Models"

Philip Lippmann, Konrad Skublicki, Joshua Tanner, Shonosuke Ishiwatari, Jie Yang

![manga image](assets/modality_comparison_short.png)
&copy; Kira Ito


## Setup
1. After cloning this repository, clone the following repositories in the same directory as well:
- https://github.com/sqbly/open-mantra-dataset
- https://github.com/sqbly/Manga-Text-Segmentation

2. Create a .env file and set variable definitions for GPT_API_KEY and PRINT_LOGS:
```
GPT_API_KEY = "<key>"
PRINT_LOGS = "[True/False]"
```

3. Download the model for text segmentation:
```bash
wget -O model.pkl https://github.com/juvian/Manga-Text-Segmentation/releases/download/v1.0/fold.0.-.final.refined.model.2.pkl
```

4. Install the required packages:
```bash
pip install -r requirements.txt

# Make sure CUDA is already installed in your workspace. Use the appropriate PyTorch version.
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Only needed if performing evaluation. Tested with Python 3.8.10.
pip install -r requirements_eval.txt
```

5. If performing evaluation using BLEURT:
Install according to the instructions on the [BLEURT repository](https://github.com/google-research/bleurt) and then use as described in their repository.

6. Your directory should now look like this:
```
├── Manga-Text-Segmentation
│   └──model.pkl
├── open-mantra-dataset
│   └── images
│       └── **/*.jpg
├── multimodal-manga-translation (This Repository)
└── .env
```

## Contributed Data set
We provide professional JA-PL translations of the slice-of-life manga *Love Hina* to create a data set for research purposes. We make volumes 1 and 14 available and our annotation process closely follows the existing annotations of the Japanese text. The newly contributed data set contains 400 pages and 3705 individual lines (*i.e.* speech bubbles, sound effects, etc.) split across the two volumes and is distributed as a set of images, corresponding to one image per page, and the corresponding metadata containing original and translated text, as well as their coordinates on the page.

We propose a 50:50 validation:test split for this data set, using the first volume (200 pages and 1810 lines) as the test set and the last volume (200 pages and 1895 lines) as the validation set. This decision is motivated primarily by the fact that the first volume establishes the story, providing a fairer benchmark for the long-context methods, as opposed to the last volume, which depends on unavailable context, being the 14th installment in the series.

For more details see the paper.

The data set is located in this repository at ```./LoveHina```.

## Quickstart
To run the full pipeline, execute all cells of the following notebook:
```bash
jupyter notebook full_pipeline_evaluation.ipynb
```

## Contact
This repository was created by Konrad Skublicki & Philip Lippmann. 

All correspondence: p.lippmann@tudelft.nl