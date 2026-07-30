"""Run OL-MomentRetrieval inference for one video and text query."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lighthouse.models import (
    OLMomentDETRPredictor,
    OLQDDETRPredictor,
    OLTaskWeavePredictor,
)

PREDICTORS = {
    "ol_moment_detr": OLMomentDETRPredictor,
    "ol_qd_detr": OLQDDETRPredictor,
    "ol_taskweave": OLTaskWeavePredictor,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PREDICTORS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--feature", default="clip", choices=["clip", "clip_slowfast", "clip_slowfast_pann", "resnet_glove"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slowfast-path", type=Path)
    parser.add_argument("--pann-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = PREDICTORS[args.model](
        ckpt_path=str(args.checkpoint),
        device=args.device,
        feature_name=args.feature,
        slowfast_path=str(args.slowfast_path) if args.slowfast_path else None,
        pann_path=str(args.pann_path) if args.pann_path else None,
    )
    predictor.encode_video(str(args.video))
    result = predictor.predict(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
