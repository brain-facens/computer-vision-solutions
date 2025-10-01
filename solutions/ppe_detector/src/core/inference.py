import cv2 as cv
from ultralytics import YOLO

def inference(source_path: str, model: str, show: bool, save: bool, conf: float, classes = list) -> None:
    detector = YOLO(model)
    class_names = detector.names

    if show:
        cap = cv.VideoCapture(source_path)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = detector(frame)
                annotated_frame = results[0].plot()

                cv.imshow("PPE Detector", annotated_frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        detections = detector.predict(
            source = source_path, 
            save = save,
            conf = conf,
            classes = classes
        )


if __name__ == "__main__":
    images = [
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/images/asphalt-3431322_1920.jpg",
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/images/construction-8734283_1920.png",
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/images/men-6523358_1920.jpg",
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/images/nlmk-9824309.jpg",
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/images/worker-4896064_1920.jpg",
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/images/worker-8959723_1920.jpg"
    ]

    videos = [
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/videos/8853463-hd_1920_1080_24fps.mp4",
        "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/videos/10151854-hd_1920_1080_24fps.mp4"
    ]

    inference(
        source_path = "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/videos/8853463-hd_1920_1080_24fps.mp4",
        model = "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/src/models/src/weights/ppe_detector_v0.pt",
        show = False,
        save = True,
        conf = 0.2, 
        classes = [3]
    )