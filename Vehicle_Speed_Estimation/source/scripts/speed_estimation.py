import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolo11n-seg.pt")
names = model.model.names

#Testes

cap = cv2.VideoCapture("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/Diurno/videoDiaPam07082025.mp4")

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/results_.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speed_obj = solutions.InstanceSegmentation(
    model = model,
    show = True,
    classes = [2, 3, 5, 7],
    conf = 0.25

)

try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        tracks = model.track(im0, persist=True, show=False)
        result = speed_obj(im0, tracks)
        video_writer.write(result.plot_im)
except KeyboardInterrupt:
    print("Finalizando")

finally:

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()