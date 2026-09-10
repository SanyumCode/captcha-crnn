"""Multi-threaded TorchScript batch inference."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .config import Config
from .dataset import IMAGE_SUFFIXES
from .utils import decode_predictions, preprocess_image


class BatchInference:
    def __init__(self, model_path: Path, threads: int = 4) -> None:
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.threads = threads

    def predict(self, path: Path) -> dict:
        started = time.perf_counter()
        with torch.no_grad():
            prediction = decode_predictions(self.model(preprocess_image(path)))[0]
        return {"file": str(path), "prediction": prediction, "seconds": time.perf_counter() - started}

    def run(self, paths: list[Path]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            return list(pool.map(self.predict, paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="批量 TorchScript 推理")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=Config.TORCHSCRIPT_PATH)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("inference_results.json"))
    args = parser.parse_args()
    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    started = time.perf_counter()
    results = BatchInference(args.model, args.threads).run(paths)
    elapsed = time.perf_counter() - started
    payload = {"count": len(results), "seconds": elapsed, "images_per_second": len(results) / elapsed if elapsed else 0, "results": results}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(results)} images, {payload['images_per_second']:.2f} img/s -> {args.output}")


if __name__ == "__main__":
    main()
