# Text-to-Video: Human Motion in Dynamic Environments

A two-phase research project for **AIE418 (Selected Topics)** that generates short video clips from text using diffusion models, evaluates their quality, and applies targeted enhancements to improve temporal consistency.

**Theme:** Human motion in dynamic environments (walking, crowded streets, human interactions)

**Pipeline:** Stable Diffusion XL (text-to-image) → Stable Video Diffusion (image-to-video)

---

## Overview

| Phase | Focus | Main artifact |
|-------|--------|---------------|
| **Phase 1** | Baseline pipeline, structured experiments, weakness analysis | `phase_01/selected-topics-projec-phase1.ipynb` |
| **Phase 2** | Semantic and temporal enhancements, ablation study, optional TTS | `phase_02/selected-topics-projec-phase02.ipynb` |

Phase 1 established that generated videos achieve strong **semantic alignment** (CLIP-SIM) but often suffer from **temporal inconsistency** (flicker, low SSIM) in complex scenes. Phase 2 addresses this with prompt conditioning and temporal stabilization, validated through an ablation study.

---

## Repository Structure

```
.
├── phase_01/
│   ├── selected-topics-projec-phase1.ipynb    # Phase 1 experiments notebook
│   └── phase_01_results/
│       └── working/experiment_results/        # Experiment outputs & metadata
│           ├── Exp_A_Guidance/                # Guidance scale ablation (5.0, 7.5, 10.0)
│           ├── Exp_B_Steps/                   # Inference steps ablation (15, 25, 40)
│           ├── Exp_C_Complexity/              # Prompt complexity (simple, medium, complex)
│           └── experiment_summary.json
│
├── phase_02/
│   ├── selected-topics-projec-phase02.ipynb   # Phase 2 enhancement notebook
│   └── phase_02_results/
│       └── working/phase2_results/
│           ├── baseline/
│           ├── enhancement_A_prompt_cfg/
│           ├── enhancement_B_temporal/
│           ├── combined_A_B/
│           ├── bonus_text_to_audio/
│           ├── phase2_ablation_table.csv
│           ├── phase2_ablation_summary.json
│           ├── phase2_paper_notes.md
│           └── repository_files/
│               ├── generate_phase2_video.py   # Standalone generation script
│               └── requirements.txt
│
└── README.md
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (notebooks were run on Kaggle with NVIDIA T4)
- ~10+ GB disk space for model weights (downloaded automatically from Hugging Face)

### Python Dependencies

Install from the bundled requirements file:

```bash
pip install -r phase_02/phase_02_results/working/phase2_results/repository_files/requirements.txt
```

Core packages:

- `torch`, `diffusers`, `transformers`, `accelerate`
- `imageio[ffmpeg]`, `opencv-python`, `pillow`
- `scikit-image`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- `lpips`, `torchmetrics`, `tqdm`

### Models Used

| Model | Purpose |
|-------|---------|
| [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Text-to-image conditioning frame |
| [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) | Image-to-video animation |
| [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | CLIP-SIM semantic evaluation |
| [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) | Bonus neural text-to-speech narration |

---

## Getting Started

### Option 1: Run the Jupyter Notebooks

1. Clone or download this repository.
2. Install dependencies (see above).
3. Open and run the notebooks in order:
   - `phase_01/selected-topics-projec-phase1.ipynb` — baseline experiments
   - `phase_02/selected-topics-projec-phase02.ipynb` — enhancements and ablation

The notebooks are designed for GPU environments (e.g., Kaggle, Google Colab, or a local CUDA machine). On first run, model weights are downloaded from Hugging Face.

### Option 2: Standalone Phase 2 Script

A compact script reproduces the combined Phase 2 pipeline (semantic conditioning + tuned SVD parameters):

```bash
cd phase_02/phase_02_results/working/phase2_results/repository_files
python generate_phase2_video.py
```

Outputs are written to `phase2_final_video/`:

- `phase2_initial_image.png` — SDXL conditioning frame
- `phase2_final_video.mp4` — generated video

---

## Phase 1: Experiments

Phase 1 runs three structured experiment groups on the SDXL + SVD pipeline:

| Experiment | Variable | Values tested |
|------------|----------|---------------|
| **A — Guidance** | Classifier-free guidance scale (T2I) | 5.0, 7.5, 10.0 |
| **B — Steps** | I2V inference steps | 15, 25, 40 |
| **C — Complexity** | Prompt scene complexity | simple, medium, complex |

### Key Phase 1 Finding

Complex dynamic scenes (e.g., crowded markets) showed the weakest temporal consistency:

- **Complex scene:** mean SSIM ≈ 0.59, flickering detected
- **Simple scene:** mean SSIM ≈ 0.80, moderate consistency
- **Semantic alignment** remained good across all runs (CLIP-SIM > 28)

Full metrics are in `phase_01/phase_01_results/working/experiment_results/experiment_summary.json`.

---

## Phase 2: Enhancements

Two complementary enhancements were proposed and evaluated individually and in combination.

### Enhancement A — Prompt Conditioning & CFG

- Richer positive prompt with anatomy and consistency constraints
- Negative prompt targeting flicker, distortion, and artifacts
- Higher guidance scale (10.0 vs. 7.5 baseline)

### Enhancement B — Temporal Stabilization

- Lower SVD `motion_bucket_id` (80) and `noise_aug_strength` (0.04)
- Temporal low-pass smoothing: `I_smooth[t] = α·I[t] + (1−α)·I_smooth[t−1]`
- Frame interpolation to reach ≥ 4 s duration at 12 fps

### Ablation Results

| Variant | CLIP-SIM | Mean SSIM | Mean Frame Diff | Duration (s) | Temporal |
|---------|----------|-----------|-----------------|--------------|----------|
| baseline | 34.22 | 0.705 | 19.999 | 3.57 | moderate |
| enhancement_A_prompt_cfg | 33.80 | 0.806 | 10.172 | 3.57 | moderate |
| enhancement_B_temporal | 34.95 | 0.957 | 4.496 | 4.08 | good |
| **combined_A_B** | **33.65** | **0.947** | **3.954** | **4.08** | **good** |

Compared to baseline, the combined variant improved mean SSIM by **+0.24** and reduced frame difference by **~16**, while maintaining good semantic alignment.

Detailed notes: `phase_02/phase_02_results/working/phase2_results/phase2_paper_notes.md`

---

## Evaluation Metrics

| Metric | What it measures |
|--------|------------------|
| **CLIP-SIM** | Text–video semantic alignment (higher is better) |
| **Mean SSIM** | Frame-to-frame structural similarity (higher = more stable) |
| **Mean Frame Diff** | Average pixel change between consecutive frames (lower = less flicker) |
| **LPIPS (temporal)** | Perceptual distance across frames (Phase 2) |
| **PSNR** | Signal quality between frames (Phase 2) |
| **FVD** | Fréchet Video Distance — placeholder in Phase 1 (requires reference dataset) |

Automated **weakness heuristics** flag temporal consistency, flickering, and semantic alignment based on metric thresholds.

---

## Bonus: Text-to-Audio Module

Phase 2 includes an optional narration layer using neural TTS:

- **Model:** SpeechT5 + HiFi-GAN vocoder
- **Features:** Multi-voice support, emotion/style presets (neutral, happy, calm), context-aware chunking
- **Output:** Synchronized audio mixed into the final video via ffmpeg

See `phase_02/phase_02_results/working/phase2_results/bonus_text_to_audio/bonus_text_to_audio_notes.md` for details.

---

## Results & Artifacts

Each experiment run saves:

- `output_video.mp4` — generated video
- `frames/` — individual frame PNGs
- `metadata.json` — prompt, parameters, metrics, and weakness analysis

Aggregated summaries:

- Phase 1: `experiment_summary.json`
- Phase 2: `phase2_ablation_summary.json`, `phase2_ablation_table.csv`
- User study template: `phase_02/.../user_study_template.csv`

---

## Limitations

- Temporal smoothing reduces flicker but may blur fast motion or reduce motion realism.
- FVD was not fully computed; motion realism requires manual visual review.
- Generation is compute-intensive (~6–9 minutes per video on a T4 GPU).
- Results are sensitive to seed and prompt wording.

---

## License

This project uses open-source models and libraries subject to their respective licenses (Stability AI, Hugging Face, Microsoft SpeechT5, etc.). Refer to each model's Hugging Face page for terms of use.
#   T e x t - t o - V i d e o - H u m a n - M o t i o n - i n - D y n a m i c - E n v i r o n m e n t s  
 