# Multimodal Emotion and Sentiment Description Generation

This repository contains the implementation of a multimodal framework that integrates visual, acoustic, and textual features from the **CMU-MOSEI** dataset to generate rich, interpretable natural language descriptions of human emotional states using **GPT-4o-mini**.

---

## Abstract
Conventional multimodal sentiment analysis often relies on predicting numerical scores (e.g., -3 to +3) or discrete emotion categories (e.g., happy, sad). While effective, these outputs lack qualitative interpretability regarding *how* an emotion is physically expressed across modalities. We propose a unified pipeline that processes multi-channel signals—facial movements (FACET/OpenFace), acoustic features (COVAREP), and textual embeddings (GloVe)—and calculates modality energy metrics ($L_2$-norms). These features are converted into structured prompts to guide GPT-4o-mini, which synthesizes low-level metrics into a coherent, single-sentence description capturing facial expressions, vocal tone, emotion intensity, and spoken topic.


## Motivation
* **Beyond Categorical Labels:** Scalar scores fail to capture complex or mixed human emotions, such as a speaker putting on a "strained smile" with a "lively tone" to express sarcasm.
* **Modality Scale Discrepancies:** Raw feature norms vary drastically across streams (e.g., Audio norm $\approx$ 100–300 vs. Text norm $\approx$ 2–3), making naive feature concatenation ineffective for qualitative interpretation.
* **LLM Contextual Synthesis:** Leveraging Large Language Models (LLMs) bridges the gap between signal-level multimodal extraction and high-level cognitive, human-interpretable summaries.


## Model Architecture

The framework consists of three sequential modules:

```
+-----------------------------------------------------------------------------------+
|                            Multimodal Inputs (CMU-MOSEI)                          |
|  [Visual: FACET/OpenFace]      [Acoustic: COVAREP]       [Language: GloVe/Text]   |
+--------------------------+-------------------+------------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|                        Feature Processing & Energy Metric                         |
|  - Clean NaN / Inf values via zero-padding                                        |
|  - Compute Vector Norms (L2): Face Energy ||F||, Audio Energy ||A||, Text Energy  |
|  - Extract 7-D Label Vector: [Sentiment, Happy, Sad, Anger, Surprise, Disgust, Fear]|
+------------------------------------+----------------------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|                       Structured Prompting & LLM Synthesis                        |
|  - Filter active emotions (threshold > 0.6) and clean transcript text              |
|  - Construct structured prompt for GPT-4o-mini (Temp: 0.6, Max Tokens: 80)         |
+------------------------------------+----------------------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|                 Output: Single-Sentence Interpretable Description                 |
+-----------------------------------------------------------------------------------+
```

* **Feature Extraction & Normalization:** Cleans missing/invalid values and extracts continuous sentiment scores and 6 Ekman emotion intensities.
* **Modality Energy Quantification:** Computes L2-norms (`E_face = ||F||_2`, `E_audio = ||A||_2`, `E_text = ||T||_2`) to represent cross-modal signal magnitude.
* **Structured Synthesis Engine:** Maps active emotion categories and modality energy metrics into a structured prompt, guiding `gpt-4o-mini` to output a unified description.

## Evaluation & Experimental Results

### Modality Feature Energy Distribution
Extracted feature norm distributions across CMU-MOSEI samples:
* **Acoustic Energy ($E_{\text{audio}}$):** 110.0 – 300.3 (Reflects pitch, dynamics, and intensity)
* **Visual Energy ($E_{\text{face}}$):** 6.6 – 20.6 (Captures facial action units and landmark shifts)
* **Textual Energy ($E_{\text{text}}$):** 2.0 – 3.1 (Normalized semantic embeddings)

### Qualitative Case Studies

| Sample ID | Ground Truth Label / Metrics | Generated Natural Language Description |
| :--- | :--- | :--- |
| **`--qXJuDtHPw`** | **Sentiment:** +1.0, **Happy:** 0.67<br>**Norms:** $E_F=10.3, E_A=114.5$ | *"The speaker exudes a joyful enthusiasm, their face bright with a wide smile and animated expressions, as they passionately discuss the distinct roles of authors, writers, and storytellers with an energetic tone."* |
| **`-3g5yACwYnA`** | **Sentiment:** +1.0, **Happy/Sad/Fear:** 0.67<br>**Norms:** $E_F=20.6, E_A=117.6$ | *"Raj Shah, with a warm yet slightly uncertain smile and an energetic tone, conveys a mix of happiness, sadness, and fear as he discusses technical expertise in adhesive operations."* |
| **`-9YyBTjo1zo`** | **Sentiment:** -1.0 (No dominant single label)<br>**Norms:** $E_F=6.6, E_A=212.9$ | *"The speaker, displaying a slightly strained smile and a lively tone, conveys a sense of sarcasm and frustration as they critique the absurdities surrounding a historically unpopular president."* |

### Key Findings
* **Cross-Modal Integration:** The model successfully resolves conflicting signal inputs (e.g., high vocal energy combined with negative sentiment) into nuanced descriptions like "sarcasm" or "strained smile."
* **Strict Constraint Adherence:** Achieved 100% compliance with single-sentence output constraints while maintaining contextual richness across facial cues, voice tone, and spoken content.

---

## 💾 Dataset Setup

This project uses the **CMU-MOSEI** (CMU Multimodal Opinion Sentiment and Emotion Intensity) Dataset for multimodal research and natural language integration.

### Manual Download

You can download the dataset directly from Zenodo:
* **Dataset Link:** [Zenodo - CMU-MOSEI Dataset](https://zenodo.org/records/18668043)

### Expected Directory Structure

After downloading, extract the files and place them into the `data/cmu_mosei/` directory as follows:

```text
3D-Human-Action-Recognition-with-Natural-Language/
├── data/
│   └── cmu_mosei/
│       ├── Audio/
│       ├── Language/
│       ├── Visual/
│       └── labels/
└── ...
