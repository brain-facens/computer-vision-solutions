import cv2

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
    line_points = [(900, 1000), (1200, 300)]

    counter = solutions.ObjectCounter(
        show = True, 
        region = line_points, 
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
        
        # cv2.imshow("PPE Detector", results.plot_im)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    count_specific_classes(
        video_path = "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/data/videos/cctv_store.mp4",
        output_video_path = "/home/nata-brain/Documents/ws/computer-vision-solutions/solutions/ppe_detector/runs/detect/video_survillance",
        model_path = 'yolo11l.pt',
        classes_to_count = [0]
    )