from ultralytics import YOLO
import yaml
import os
import argparse

cur_dir = os.path.dirname(os.path.realpath(__file__))


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument(
        "--config", default=os.path.join(cur_dir, "train_configs.yaml"), type=str
    )
    args = parser.parse_args()
    return args

def train(cfg):
    model = YOLO(cfg["model_path"], verbose=False)
    results = model.train(
        data=cfg["data_path"],
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        device="0",
        cache=cfg["cache"],
        batch=cfg["batch"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        shear=cfg["shear"],
        fliplr=cfg["fliplr"],
    )
    return results


def validate(cfg, best_model_path):
    model = YOLO(best_model_path)
    results = model.val(data=cfg["data_path"], imgsz=cfg["imgsz"], device="0")
    return results


def main(cfg):
    results = train(cfg)
    run_path = results.save_dir
    print(f"Results saved to {run_path}")
    # best_model_path = os.path.join(run_path, "weights", "best.pt")
    # val_results = validate(cfg, best_model_path)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)
