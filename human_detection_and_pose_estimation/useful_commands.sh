# export the yolo model to be compilable for the IMX500 chip
yolo export model=yolov8n_COCO.pt format=imx int8=True imgsz=320

# convert the exported zip file to rpk binary on RPI
imx500-package -i yolov8n_COCO_imx_model/packerOut.zip -o yolov8n_COCO_rpk

# rename the rpk file
mv yolov8n_COCO_rpk/network.rpk yolov8n_COCO.rpk

# stream the video from RPI
gst-launch-1.0 tcpclientsrc host=10.0.1.37 port=10001 ! h264parse ! avdec_h264  ! autovideosink
