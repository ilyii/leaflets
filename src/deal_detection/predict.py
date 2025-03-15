from ultralytics import YOLO
import argparse
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))


def predict(model_path, input_path, output_path, imgsz, conf, iou):
    """Run predictions on the input data."""
    model = YOLO(model_path)

    results = model.predict(
        source=input_path,  # Path to input image or directory
        imgsz=imgsz,  # Image size
        conf=conf,  # Confidence threshold
        iou=iou,  # IOU threshold
        device="0",  # GPU device
        save=True,  # Save output images
        save_txt=True,  # Save predictions in text files
        save_conf=True,  # Save confidences in text files
        project=output_path,  # Directory to save results
        name="predictions"  # Subdirectory name
    )
    return results


def main():
    """Main function to execute prediction pipeline."""
    parser = argparse.ArgumentParser(description="YOLO Prediction Script")
    parser.add_argument("--weights", "-w", type=str, required=True, help="Path to the trained YOLO model")
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Path to input images or directory")
    parser.add_argument("--output-path", "-o", type=str, required=True, help="Directory to save prediction results")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.95, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.6, help="IOU threshold (default: 0.45)")

    args = parser.parse_args()

    results = predict(
        model_path=args.weights,
        input_path=args.input_path,
        output_path=args.output_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou
    )

    print("Prediction complete.")
    print(f"Results saved to: {os.path.join(args.output_path, 'predictions')}")


if __name__ == "__main__":
    main()
