
import cv2
from detector import detect_defects

image_path = input("Enter image path: ").strip().strip('"')

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found.")
else:
    result = detect_defects(image)

    print("Total defects detected:", result["count"])

    for defect in result["detections"]:
        print(defect["defect"], "-", defect["confidence"])

    cv2.imwrite("detected_output.jpg", result["annotated_image"])
    print("Output saved as detected_output.jpg")
