from ultralytics import YOLO
import yaml
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(cfg):
    model = YOLO(cfg["model_path"])
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
    best_model_path = os.path.join(run_path, "weights", "best.pt")
    val_results = validate(cfg, best_model_path)
    print(val_results)


if __name__ == "__main__":
    cfg = load_config(os.path.join(cur_dir, "train_configs.yaml"))
    main(cfg)
