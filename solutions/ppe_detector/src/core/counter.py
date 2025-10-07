import cv2
import datetime
from ultralytics import solutions

def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS)
    )

    video_writer = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (w, h)
    )

    # (x, y)
    line_points_vehicle = [(10, 300), (1270, 300)]

    counter = solutions.ObjectCounter(
        show = True, 
        region = line_points_vehicle, 
        model = model_path, 
        classes = classes_to_count, 
        tracker = 'bytetrack.yaml'
    )

    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = counter(im0)
        # if results.classwise_count:
        #     current_datetime = datetime.datetime.now()
        #     if results.classwise_count['car']:
        #         print(f"Contagem de Carros: {results.classwise_count['car']} {current_datetime}")
        #     elif results.classwise_count['motorcycle']:
        #         print(f"Contagem de Motos: {results.classwise_count['motorcycle']} {current_datetime}")
        #     elif results.classwise_count['bus']:
        #         print(f"Contagem de Onibus: {results.classwise_count['bus']} {current_datetime}")
        #     elif results.classwise_count['truck']:
        #         print(f"Contagem de Caminhões: {results.classwise_count['truck']} {current_datetime}")
        #     elif results.classwise_count['truck']:
        #         print(f"Contagem de Bicicletas: {results.classwise_count['bicycle']} {current_datetime}")
            
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    count_specific_classes(
        video_path = "rtsp://admin:Smart2022@172.16.231.120:50000/video",
        output_video_path = "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/runs/detect/video_survillance",
        model_path = 'yolo11l.pt',
        classes_to_count = [0, 1, 2, 3, 5, 7]
    )