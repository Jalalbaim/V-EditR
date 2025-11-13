# V-EditR

**V-EditR** — A reasoning-first image editor powered by Vision–Language Models for intelligent, context-aware image manipulation.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

</div>

## 🎯 Overview

V-EditR is an advanced image editing pipeline that understands natural language instructions and applies precise edits based on semantic reasoning. Unlike traditional image editors that rely purely on style cues, V-EditR analyzes scene context, object relationships, and spatial information before making modifications.

### Why V-EditR?

Modern image editing tools often fail on requests that require understanding relations, counts, or context. For example:

- ❌ "Add two red cars next to the blue truck." → Adds cars in wrong locations
- ❌ "Make the person holding the phone wear a black jacket." → Modifies wrong person
- ❌ "Remove the chair behind the table." → Removes wrong object

V-EditR solves these problems by:

✅ **Understanding Context** — Interprets free-form instructions with semantic reasoning  
✅ **Spatial Reasoning** — Handles relations like "next to", "behind", "holding"  
✅ **Object Grounding** — Precisely locates objects using GroundingDINO + SAM  
✅ **Smart Editing** — Applies targeted modifications without disturbing unrelated content

## 🏗️ Architecture

V-EditR operates through a multi-stage pipeline:

```
┌─────────────────┐
│ Text Instruction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Plan Generator │ ───► │ Operation(s) │
│  (LLM Parser)   │      │ + Targets    │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Object Grounding    │
                    │  • GroundingDINO     │
                    │  • SAM (Segment)     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Edit Application   │
                    │  • InstructPix2Pix   │
                    │  • Add-It/ControlNet │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Edited Image       │
                    └──────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (recommended for GPU acceleration)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for optimal performance)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Jalalbaim/V-EditR.git
   cd V-EditR
   ```

2. **Create a virtual environment**

3. **Install dependencies**

   ```bash
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # For CPU only
   pip install torch torchvision torchaudio

   # Install other requirements
   pip install requirements.txt
   ```

4. **Download model weights**

   Create a `weights/` directory and download the following:

   - **GroundingDINO**: [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases)
   - **SAM**: [sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything#model-checkpoints)

5. **Configure paths**

   Edit `configs/grounding.yaml` to update checkpoint paths:

   ```yaml
   grounding:
     dino:
       ckpt: "path/to/your/groundingdino_swint_ogc.pth"
     sam:
       ckpt: "path/to/your/sam_vit_h_4b8939.pth"
   ```

## 📖 Usage

### Basic Command

```powershell
python scripts/run_edit.py --image assets/sample.jpeg --instruction "add a red car next to the truck" --tag my_edit
```

### Parameters

- `--image`: Path to input image (required)
- `--instruction`: Text instruction describing the edit (required)
- `--tag`: Custom tag for the output folder (default: "phase4")

### Examples

```powershell
# Add object
python scripts/run_edit.py --image assets/sample.jpeg --instruction "add two blue cars on the road"

# Remove object
python scripts/run_edit.py --image assets/sample.jpeg --instruction "remove the truck"

# Modify attributes
python scripts/run_edit.py --image assets/sample.jpeg --instruction "make the truck red"

# Complex relations
python scripts/run_edit.py --image assets/sample.jpeg --instruction "add a person next to the car holding an umbrella"
```

### Output Structure

Results are saved in `runs/TIMESTAMP_TAG/`:

```
runs/
└── 20251113_143752_my_edit/
    ├── run_summary.json        # Execution metadata
    └── artifacts/
        ├── input.jpg           # Original image
        ├── edited.jpg          # Final result
        ├── plan.json           # Generated action plan
        ├── grounding.json      # Object detection results
        ├── boxes_*.jpg         # Bounding box visualizations
        ├── masks_*.jpg         # Segmentation masks
        ├── validator.json      # Validation report
        └── verifier.json       # Verification results
```

### Project Structure

```
V-EditR/
├── configs/              # Configuration files
├── src/
│   ├── editors/          # Image editing modules
│   ├── grounding/        # Object detection & segmentation
│   ├── planners/         # Instruction parsing
│   ├── validators/       # Output validation
│   └── verifiers/        # Result verification
├── scripts/              # Execution scripts
├── tests/                # Unit tests
├── weights/              # Model checkpoints
└── runs/                 # Output directory
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Open-set object detection
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Instance segmentation
- [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/) - Instruction-based editing
- [Stable Diffusion](https://stability.ai/) - Generative models

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

**Author**: Jalal Baim  
**Repository**: [https://github.com/Jalalbaim/V-EditR](https://github.com/Jalalbaim/V-EditR) — a reasoning-first image editor powered by a Vision–Language Model.

---
