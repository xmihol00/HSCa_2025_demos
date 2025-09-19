from ultralytics import YOLO

model = YOLO("yolov8s_COCO.pt")
results = model.train(
    data="/mnt/matylda5/xmihol00/yolov8/datasets/vehicles/data.yaml", 
    seed=42, epochs=500, batch=48, imgsz=(640,640), lr0=0.01, lrf=0.001, amp=False, save_period=10, pretrained=True
)
model.eval()