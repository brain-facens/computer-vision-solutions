import time

import cv2
from ultralytics import YOLO

# Modelos
model_car = YOLO("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/yolo11n-seg.pt")
model_wheels = YOLO("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/yolov11n_wheels.pt")

# Vídeo de entrada e saída
cap = cv2.VideoCapture("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/Diurno/videoDiaPam06082025.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    "/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/results_.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Classes
classes = {0: "Roda", 2: "Carro", 3: "Moto", 5: "Onibus", 7: "Caminhao"}

# isegment = solutions.InstanceSegmentation(
#     show=True,
#     model=model_car,
#     conf=0.6,
#     classes=[5, 7],
#     tracker="/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/source/scripts/bytetrack.yaml"
# )

time_start = time.time()
try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        # car_results = isegment(im0)

        # print(solutions.InstanceSegmentation.display_output)
#===
        #Rastreamento de veículos
        car_results = model_car.track(im0, classes=[2, 3, 5, 7], conf=0.50, iou = 0.7, persist=True, tracker="/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/source/scripts/bytetrack.yaml", show_labels=True, device="cuda")

        for result in car_results:
            for box_car in result.boxes:
                x1_car, y1_car, x2_car, y2_car = map(int, box_car.xyxy[0])
                veiculo = classes[box_car.cls.item()]
                conf_veiculo = round(box_car.conf.item(), 2)

        
                cv2.rectangle(im0, (x1_car, y1_car), (x2_car, y2_car), (255, 0, 0), 2)
            
                # Contagem de rodas por tipo
                if veiculo == "Carro":
                    count = 4
                elif veiculo == "Moto":
                    count = 2 
                elif veiculo == "Onibus" or veiculo == "Caminhao":
                    # Crop do veículo para detectar rodas
                    cropped_image = im0[y1_car:y2_car, x1_car:x2_car]
                    wheel_crop_results = model_wheels.predict(cropped_image, conf=0.20, iou = 0.7, device="cuda")

                    count = 0
                    for w_result in wheel_crop_results:
                        for box in w_result.boxes:
                            if classes[box.cls.item()] == "Roda":
                                count += 1
                                x1_w, y1_w, x2_w, y2_w = map(int, box.xyxy[0])
                                x1_w += x1_car
                                x2_w += x1_car
                                y1_w += y1_car
                                y2_w += y1_car
                                cv2.rectangle(im0, (x1_w, y1_w), (x2_w, y2_w), (0, 255, 0), 2)
                
                
                texto_carro = f"{veiculo} / conf: {conf_veiculo} / {count}"
                cv2.putText(im0, texto_carro, (x1_car, y1_car), fontScale=1, fontFace=1, color=(0, 0, 255), thickness=2)

#===
        video_writer.write(im0)
        cv2.imshow("Resultado", im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Ocorreu um erro: {e}")

finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    time_end = time.time()
    print("Processamento: ")
    print(time_end - time_start)
