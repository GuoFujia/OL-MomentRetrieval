# OL-MomentRetrieval

Official inference implementation for online video moment retrieval with three architectures:

- **OL-MomentDETR**
- **OL-QDDETR**
- **OL-TaskWeave**

The repository contains model definitions, feature extractors, and a single-video inference interface. It intentionally does not include training code, datasets, experiment logs, or evaluation artifacts.

## Installation

Create a Python environment (Python 3.9+ recommended), install PyTorch for your CUDA or CPU platform, then install this package:

```bash
pip install -e .
```

Some feature backbones require their own pretrained assets. Supply their local paths with `--slowfast-path` and `--pann-path` when applicable.

## Checkpoints

Model checkpoints are not committed to this repository. Download the official checkpoint for the model and feature configuration you intend to use, then pass its local path to the command below. A checkpoint must contain the saved `opt` configuration and `model` state dictionary expected by `lighthouse.models`.

## Run inference

```bash
python api_example/run_inference.py \
  --model ol_moment_detr \
  --checkpoint /path/to/model.ckpt \
  --video /path/to/video.mp4 \
  --query "A person opens a door" \
  --feature clip \
  --device cuda
```

The command writes JSON containing ranked temporal windows (`start`, `end`, `score`) and per-clip saliency scores. Use `--help` to see all supported feature choices.

## Python API

```python
from lighthouse.models import OLMomentDETRPredictor

predictor = OLMomentDETRPredictor(
    ckpt_path="/path/to/model.ckpt",
    device="cuda",
    feature_name="clip",
)
predictor.encode_video("/path/to/video.mp4")
result = predictor.predict("A person opens a door")
```

Replace `OLMomentDETRPredictor` with `OLQDDETRPredictor` or `OLTaskWeavePredictor` to use the other official models.

## Repository layout

- `lighthouse/common/`: online retrieval architectures and shared model components
- `lighthouse/feature_extractor/`: video, text, and audio feature extraction
- `lighthouse/frame_loaders/`: video decoding utilities
- `api_example/run_inference.py`: command-line inference entry point
- `configs/model/`: architecture identifiers for checkpointed models

## License and acknowledgements

This project is released under the Apache License 2.0. It incorporates components derived from Moment-DETR and the feature-backbone projects referenced in the source files; please follow their respective licenses when using those components.
