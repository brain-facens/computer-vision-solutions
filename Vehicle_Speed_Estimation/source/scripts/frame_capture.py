import cv2
import os

from ultralytics import YOLO

model_car = YOLO("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/yolo11x.pt")

# Caminho da pasta onde você quer salvar
output_dir = "/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/dataset_frames_rodovia"

# Cria a pasta se não existir
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/Diurno/melhor_cenario.webm")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    car_results = model_car.track(frame, classes=[5, 7], conf=0.90, iou = 0.7, persist=True, show_labels=True, device="cuda")
    if car_results and len(car_results[0].boxes) > 0:
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
    # filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    # cv2.imwrite(filename, frame)
    frame_count += 1

cap.release()
