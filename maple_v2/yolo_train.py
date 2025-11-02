from ultralytics import YOLO
import os



def model_train():
    model = YOLO('yolo11n.pt')
    model.train(
        data='my_train.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda',
        patience=30,
        workers=4
    )

def model_predict(source=r'C:\Repo\D4\final\val\images', save=True):
    """Predict on a single image, a directory of images, or a glob pattern.

    - source: file path, directory path, or glob pattern (e.g. 'images/*.jpg').
    - save: whether to save annotated results.
    - project: output directory base for saving (passed to ultralytics predict).
    - name: subfolder name under project for this run.

    If source is a directory, ultralytics' predict can accept the directory directly.
    """
    model = YOLO(r'C:\Repo\D4\final\runs\detect\train\weights\best.pt')

    if os.path.isdir(source):
        print(f"Predicting on directory: {source}")
        results = model.predict(source=source, save=save, exist_ok=True)
    elif os.path.isfile(source):
        print(f"Predicting on file: {source}")
        results = model.predict(source, save=save, exist_ok=True)
    else:
        # treat as pattern or maybe a single file path string
        print(f"Predicting on source: {source}")
        results = model.predict(source, save=save, exist_ok=True)

    return results


if __name__ == '__main__':
    # model_train()
    src = r"C:\Repo\D4\final\train\images\2.jpg"
    res = model_predict(src)
    print(res[0].boxes)
