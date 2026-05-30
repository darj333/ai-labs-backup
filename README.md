# AI Labs — Summary

Each lab is implemented as a Jupyter notebook with exercises, unit tests, and visualizations where applicable.

---

## Lab 1 — Algorithms & Data Structures

**Notebook:** `lab1.ipynb`

**Goal:** Practice fundamental algorithms and analyze their time complexity using pure Python.

---

## Lab 2 — Data Exploration, Image & Text Processing, Normalization

**Notebook:** `lab2.ipynb`

**Goal:** Explore the Kaggle Data Science Survey, process images and Romanian text, and apply normalization techniques across tabular, image, and text data.

### What was done

#### Block 1 — Kaggle Data Science Survey

| Area | Tasks |
|------|--------|
| Loading & utilities | `loadDataMoreInputs` (stdlib `csv`), `loadData` / `pd.read_csv`, `extractFeature`, `plotDataHistogram` |
| Respondent metadata | `nrResp` — count respondents; `extract_resp_data` — one respondent's attributes; `nr_tip` — column count/types |
| Completeness | `respondenti_date_complete` — respondents who answered every question group; `respondenti_date_complete_romania_femei` — same for Romanian women |
| Education (Q4) | `media_studii_c5` — global mean years of study; `media_studii_ro` — Romania and Romania women |
| Romania + gender + languages | `analiza_femei_romania_limbaje` — Python vs C++ among Romanian women; `femei_cod_python` / `femei_cod_cpp` |
| Exploration | Per-column unique value counts and sorted distinct values for categorical fields |
| Experience (Q6) | `analiza_vechime` — descriptive stats on coding experience; `distributie_py_age` — age bar chart for Python users; filtered variants for Romania and Romania + women |
| Outliers | `outlieri_vechime` — IQR-based outliers on experience + boxplot |

#### Problema 2 — Image processing (`data/images/`)

- Display sample image (`BERT.png`)
- Batch resize (`redimensionare_simpla`) — all `.png`/`.jpg`/`.webp` to 128×128, BGR→RGB, grid plot
- Grayscale conversion and visualization
- Gaussian blur (`cv2.GaussianBlur`, 15×15) before/after comparison
- Canny edge detection (100, 200) on grayscale images

#### Problema 3 — Text processing (`data/texts.txt`)

- Read course announcement text (`citeste_text`)
- Count sentences, words, and distinct words (`nr_propozitii`, `nr_cuvinte`, `nr_cuvinte_diferite`)
- Min/max word by length (`cuvant_min_max`)
- Remove Romanian diacritics (`fara_diacritice`)
- Synonyms via Datamuse API (`sinonime_cuvant`)
- Spell check via LanguageTool API (`cuvant_corectat`)
- Collapse repeated letters (`delete_insane_letters`)
- Synonyms via web scraping on dictionardesinonime.ro (`sinonim`)
- Full pipeline: longest word → clean → spell-check → synonym lookup

#### Problema 4 — Normalization

| Sub-problem | Description |
|-------------|-------------|
| 4.1 (survey) | Map education and experience to years; apply min-max, zero-centering, and z-score normalization; histograms on study duration and coding experience |
| 4.2 (images) | Min-max and z-score normalization on `diffusionModel.jpg`, `Karpaty.jpg`, `LeCun.jpg` — side-by-side images + pixel histograms |
| 4.3 (text) | Word frequency counts; min-max, term-frequency, and log scaling — 2×2 bar charts |

### Tools

| Category | Libraries / services |
|----------|----------------------|
| Data | `pandas`, `numpy`, `csv` |
| Visualization | `matplotlib.pyplot`, `matplotlib.image` |
| Images | `opencv-python` (cv2), `os` |
| Text | `re`, `collections.Counter` |
| Web / APIs | `requests`, `BeautifulSoup` (bs4), Datamuse API, LanguageTool API, dictionardesinonime.ro |
| Math | `math.log`, `math.sqrt` |

### Data

| Path | Purpose |
|------|---------|
| `data/surveyDataSience.csv` | Kaggle Data Science Survey (~25,973 respondents, 369 columns) |
| `data/images/` | AI/ML images: e.g. `BERT.png`, `diffusionModel.jpg`, `Karpaty.jpg`, `LeCun.jpg` |
| `data/texts.txt` | Romanian course announcements |

### Key concepts

- CSV loading with pandas (`skiprows`, `low_memory=False`) and manual `csv` reader
- Survey analytics: filtering by country/gender, education→years mapping, completeness by question groups
- Descriptive statistics, bar charts, IQR outlier detection, boxplots
- Image I/O, resize, color spaces (BGR/RGB), grayscale, Gaussian blur, Canny edges
- Text tokenization, diacritic stripping, external synonym/spell APIs, web scraping
- Min-max scaling, zero-centering, z-score normalization
- Word-frequency normalization: min-max, term-frequency, log scaling

---

## Lab 3 — OCR with Azure Computer Vision

**Notebook:** `lab3.ipynb`

**Goal:** Use Azure OCR, evaluate recognition quality with string and bounding-box metrics, and experiment with image preprocessing.

### What was done

1. **Azure OCR setup** — authenticate and call `read_in_stream` on printed-text images
2. **Basic OCR evaluation** — compare OCR output lines to ground truth (exact line match count)
3. **Character-position errors** — count mismatched characters at aligned positions (`how_far_off`, `how_far_off2`)
4. **String distance metrics** — implement and compare:
   - Hamming distance
   - Levenshtein distance (library + from-scratch matrix DP)
   - Jaro / Jaro-Winkler similarity (library + custom implementation)
   - Longest Common Subsequence (LCS) via `difflib`
5. **OCR error rates** — Character Error Rate (CER) and Word Error Rate (WER)
6. **Bounding-box evaluation** — extract OCR line boxes; measure center distance and IoU vs reference boxes
7. **Confusion-aware Levenshtein** — lower penalty for commonly confused character pairs (m/n, e/l, S/L)
8. **Edge-detection preprocessing** — compare OCR quality with/without Canny edge detection; visualize results

### Tools

| Category | Libraries |
|----------|-----------|
| Cloud OCR | `azure-cognitiveservices-vision-computervision`, `msrest`, `config` (VISION_KEY, VISION_ENDPOINT) |
| String metrics | `python-Levenshtein`, `jellyfish`, `difflib` |
| Image processing | `PIL`, `opencv-python` (cv2), `matplotlib`, `numpy` |
| Other | `os`, `time`, `requests`, `re`, `math`, `io` |

### Data

| File | Purpose |
|------|---------|
| `data/test1.png` | OCR test — target: `"Google Cloud Platform"` |
| `data/test2.jpeg` | OCR test — Romanian AI lab success message |
| `data/test3.png` | Bounding-box positioning evaluation |

### Key concepts

- Azure Computer Vision Read API
- Edit distances: Levenshtein, Hamming, Jaro-Winkler, LCS
- CER / WER for OCR evaluation
- IoU and center-distance for bounding-box alignment
- Canny edge detection as OCR preprocessing

---

## Lab 4 — Azure Vision: Bicycle Detection & Evaluation

**Notebook:** `lab4.ipynb`

**Goal:** Use Azure Computer Vision object/tag detection to classify images as containing a bicycle, evaluate predictions with sklearn and hand-built metrics, draw bounding boxes, and measure localization quality with IoU.

### What was done

1. **Azure setup** — `computer_vision_client()` via `config.VISION_KEY` / `VISION_ENDPOINT`
2. **Single-image detection** — `has_bike1` — `analyze_image_in_stream` with `VisualFeatureTypes.objects` and `.tags`; detect `bicycle` in object properties or tags (confidence > 0.5)
3. **Batch detection** — `has_bike` — process all `.jpg` files in a folder; `test_has_bike()` on `data/test_images` (`bike02`–`bike09`)
4. **Evaluation v1 (sklearn)** — `evalClassificationV1` — confusion matrix, accuracy, per-class precision/recall
5. **Evaluation v2 (manual)** — `evalClassificationV2` — manual TP/TN/FP/FN, accuracy, precision, recall
6. **Evaluation with plots** — `evalClassificationV11` / `V22` with `ConfusionMatrixDisplay`; `get_labels_for_evaluation` helper
7. **Ground-truth experiment** — manual `ground_truth` dict: only `bike02.jpg` = true bike; others false — exposes false positives when Azure tags all images as bicycle
8. **Filename-based labels** — `getRealLabel` — `True` if filename starts with `"bike"`; `hasBike`, `BikeAcc` (sklearn), `myBikeAcc` (manual metrics)
9. **Bounding boxes (PIL)** — `drawBikes` — draw Azure `bicycle` rectangles; save annotated images to `output/`
10. **Bounding boxes (matplotlib)** — `bikeCoords` + `plt.Rectangle` overlays on predicted bike images
11. **Manual annotations + IoU** — `manual_annotations` dict; `compute_iou`; `evaluate_detection` — per-image best IoU vs manual boxes, mean IoU (~0.612), detection rate @0.5 (~67%)

### Tools

| Category | Libraries |
|----------|-----------|
| Cloud vision | `azure-cognitiveservices-vision-computervision`, `msrest` (`CognitiveServicesCredentials`), `VisualFeatureTypes` |
| ML metrics | `sklearn.metrics` (confusion matrix, accuracy, precision, recall, `ConfusionMatrixDisplay`) |
| Images / viz | `PIL` (`Image`, `ImageDraw`), `matplotlib.pyplot`, `numpy` |
| Other | `os`, `glob`, `config` (credentials) |

### Data

| Path | Purpose |
|------|---------|
| `data/test_images/bike02.jpg` … `bike09.jpg` | 8 test images; primary evaluation set |
| `output/` | Annotated images with drawn bicycle bounding boxes |
| `config.py` | `VISION_KEY`, `VISION_ENDPOINT` (required to run; not committed) |

### Key concepts

- Azure Analyze Image API (in-stream): object detection and image tagging
- Binary classification: bicycle vs no_bike
- Confusion matrix, accuracy, precision, recall (sklearn vs manual counting)
- Bounding box extraction (`object.rectangle`) and visualization
- Intersection over Union (IoU) for localization; mean IoU and detection rate at threshold 0.5
- Comparing API labels, filename heuristics, and hand-drawn ground truth

---

## Lab 5 — Linear Regression

**Notebooks:** `lab5.ipynb`, `lab5_date2.ipynb`, `lab5_date3.ipynb`, `lab5_corectat.ipynb`

**Goal:** Build linear regression from scratch with gradient descent, then compare with scikit-learn on World Happiness Report 2017 data.

### What was done

1. **Toy linear regression (warm-up)** — `Linear` class with manual gradient updates on synthetic points
2. **Improved gradient descent** — `Linear2` with proper batch gradients and learning rate
3. **Univariate regression** — predict `Happiness.Score` from `Family`:
   - Histograms and scatter plots
   - 80/20 train/validation split (random seed 47)
   - Train with custom `Linear` (gradient descent)
   - Plot learned model vs data
   - Validation MSE per sample
4. **Multivariate regression** — predict happiness from `Economy..GDP.per.Capita.` + `Freedom`:
   - Feature normalization (min-max)
   - Custom `Linear2feat` (3-parameter model: w₀ + w₁·GDP + w₂·Freedom)
   - Train/validate without a library
5. **Sklearn comparison** — `sklearn.linear_model.LinearRegression` for univariate and multivariate cases
6. **Corrected variant (`lab5_corectat.ipynb`)** — closed-form least squares:
   - Univariate: normal equations
   - Multivariate: matrix form **w = (XᵀX)⁻¹ XᵀY**
   - MAE and RMSE on train/validation
   - Side-by-side comparison with sklearn coefficients

### Tools

| Category | Libraries |
|----------|-----------|
| Data & numerics | `pandas`, `numpy` |
| Visualization | `matplotlib.pyplot` |
| ML | `sklearn.linear_model.LinearRegression` |
| Custom | Pure Python classes: `Linear`, `Linear2`, `Linear2feat` |

### Data

| File | Used in |
|------|---------|
| `data/lab05/v1_world-happiness-report-2017.csv` | `lab5.ipynb`, `lab5_corectat.ipynb` |
| `data/lab05/v2_world-happiness-report-2017.csv` | `lab5_date2.ipynb` |
| `data/lab05/v3_world-happiness-report-2017.csv` | `lab5_date3.ipynb` (with `dropna` on key columns) |

**Key columns:** `Family`, `Happiness.Score`, `Economy..GDP.per.Capita.`, `Freedom`

### Key concepts

- Linear regression (univariate and multivariate)
- Gradient descent and manual weight updates
- Closed-form least squares (normal equations)
- Feature normalization, train/validation split, MSE loss
- Evaluation metrics: MAE, RMSE
- Custom implementation vs sklearn comparison

---

## Lab 7 — Image Classification (Sepia vs Original)

**Notebook:** `lab7.ipynb`

**Goal:** Classify cat photos as original or sepia-filtered using neural networks — both with scikit-learn and a from-scratch implementation.

### What was done

1. **Sepia filter** — apply sepia tone matrix via OpenCV to all original images
2. **Dataset building** — label images (0 = original, 1 = sepia); save metadata to CSV
3. **MLP with sklearn** — load/resize images (32×32), flatten, normalize; train `MLPClassifier`; report accuracy (~75%)
4. **Hyperparameter study** — vary hidden layer sizes `(32,)`, `(64,)`, `(128,)`, `(64, 32)`, `(128, 64)` and `max_iter` (5, 25, 50, 100)
5. **Custom ANN (`Anna` class)** — feedforward network with ReLU hidden layers, sigmoid output, binary cross-entropy, mini-batch SGD, backpropagation; He-style weight initialization

### Tools

| Category | Libraries |
|----------|-----------|
| Data & numerics | `numpy`, `pandas` |
| Image processing | `opencv-python` (cv2), `matplotlib.pyplot` |
| ML | `sklearn.model_selection.train_test_split`, `sklearn.preprocessing.StandardScaler`, `sklearn.neural_network.MLPClassifier`, `sklearn.metrics.accuracy_score` |
| Other | `pathlib.Path`, `random`, `IPython.display` |

### Data

| Path | Description |
|------|-------------|
| `data/lab07/pisic*.jpeg` | 10 original cat images (pisic1, 3–11) |
| `data/lab07/sepia_pisic*.jpeg` | 10 sepia versions (generated in notebook) |
| `data/lab07/image_database.csv` | Metadata: filename, filepath, label, class |

**Total:** 20 images, binary classification (original vs sepia)

### Key concepts

- Image preprocessing: resize, BGR→RGB, min-max normalization, flattening
- Color transformation (sepia matrix)
- Multi-layer perceptron (MLP)
- Train/test split, feature scaling (`StandardScaler`)
- Hyperparameter tuning (architecture, iterations)
- Backpropagation, ReLU, sigmoid, binary cross-entropy
- Weight initialization (He-style scaling for ReLU)

---
