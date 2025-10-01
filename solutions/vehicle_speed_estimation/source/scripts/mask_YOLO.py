from ultralytics import YOLO
import cv2
import numpy as np
import time

model_car = YOLO("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/yolo11x-seg.pt")
model_wheels = YOLO("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/yolov11n_wheels.pt")


cap = cv2.VideoCapture("/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/Gravação de tela de 2025-08-11 15-07-34.webm")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    "/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/Vehicle_Speed_Estimation/examples/testes/result_01.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Classes
classes = {0: "Roda", 2: "Carro", 3: "Moto", 5: "Onibus", 7: "Caminhao"}

count = 0
time_start = time.time()

try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        
        #Código começa aqui

        car_results = model_car.predict(im0, classes=[2, 5, 7], conf=0.8, tracker="/home/rafael-brain/GitHub/Trabalho/computer-vision-solutions/bytetrack.yaml", show_labels=True, device="cuda")

        for result in car_results:
            if result.masks is None:
                continue

            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                x1_car, y1_car, x2_car, y2_car = map(int, box.xyxy[0])

                pts = points[0]

                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])

                offset = 25
                expand_pts = []

                for x, y in pts:
                    dx = x - cx
                    dy = y - cy

                    dist = np.sqrt(dx**2 + dy**2)

                    if dist == 0:
                        nx, ny = x, y

                    else:
                        nx = x + dx / dist * offset
                        ny = y + dy / dist * offset
                    expand_pts.append([int(nx), int(ny)])

                expand_pts = np.array([expand_pts], dtype=np.int32)

                #Identificar a classe e o confidência
                veiculo = classes[int(box.cls[0])]
                conf_veiculo = round(float(box.conf[0]), 3)
            
                #Desenhar as linhas do caminhão
                # cv2.polylines(im0, points, isClosed=True, color=(0, 0, 255), thickness=1)
                
                mask_img = np.zeros(im0.shape[:2], dtype=np.uint8)
                #mask_img.fill(128)  # preenche todo o array com 128

                cv2.fillPoly(mask_img, expand_pts, 255)

                masked_crop = cv2.bitwise_and(im0, im0, mask=mask_img)
                
                # cv2.imshow("Masked Crop", masked_crop)
                # cv2.waitKey(0)
                
                wheel_crop_results = model_wheels.track(masked_crop, conf=0.1, iou=0.7,device="cuda")
                
                for w_result in wheel_crop_results:
                    for w_box in w_result.boxes:
                        if classes[int(w_box.cls[0])] == "Roda" and veiculo in ["Caminhao", "Onibus"]:
                            count += 1
                            x1_w, y1_w, x2_w, y2_w = map(int, w_box.xyxy[0])
                            cv2.rectangle(im0, (x1_w, y1_w), (x2_w, y2_w), (0, 255, 0), 2)
                            
                cv2.polylines(im0, points, isClosed=True, color=(0, 0, 255), thickness=1)
                #Implementação da Label
                texto_carro = f"{veiculo} / conf: {conf_veiculo} / {count}"
                cv2.putText(im0, texto_carro, (x1_car, y1_car - 10), fontScale=1, fontFace=1, color=(0, 0, 255), thickness=2)
                count = 0
                # image = points
                # print(image)
                
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
