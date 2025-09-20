from ultralytics import YOLO

model = YOLO("yolov8s_COCO.pt")
results = model.train(
    data="/mnt/matylda5/xmihol00/HSCa_2025_demos/yolov8_transfer_learning/dataset/data.yaml", # TODO: Update this path as needed
    seed=42, epochs=500, batch=48, imgsz=640, lr0=0.01, lrf=0.001, amp=False, save_period=10, pretrained=True,
    device=5
)
model.eval()