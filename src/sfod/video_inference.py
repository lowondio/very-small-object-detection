from __future__ import annotations

from pathlib import Path
from typing import Any


def run_video_inference(
    source: str | Path,
    weights: str | Path,
    *,
    detector: str = "yolo",
    output_dir: str | Path = "runs/video_inference",
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.70,
    max_det: int = 300,
    imgsz: int = 640,
    save: bool = True,
    show: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    """Run a video through a trained detector and save the annotated result.

    Returns a dictionary with the output directory and path(s) of created files.
    """
    if detector != "yolo":
        raise ValueError("Video inference is currently supported for the YOLO detector family.")

    from ultralytics import YOLO

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"video not found: {source_path}")

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    out_dir = Path(output_dir)
    project = str(out_dir.parent)
    name = out_dir.name

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        save=save,
        project=project,
        name=name,
        exist_ok=True,
        show=show,
        stream=stream,
        verbose=False,
    )

    output_info: dict[str, Any] = {
        "source": str(source_path),
        "weights": str(weights_path),
        "output_dir": str(out_dir),
        "results": [],
    }

    for result in results:
        save_dir = getattr(result, "save_dir", None)
        if save_dir is not None:
            output_info["results"].append(str(Path(save_dir)))

    if not output_info["results"]:
        output_info["results"] = [str(out_dir)]

    return output_info
