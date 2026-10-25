
import av
import cv2
from streamlit_webrtc import VideoProcessorBase
from detector import detect_defects

class LiveYOLODetector(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        if image.shape[1] > 640:
            scale = 640 / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale)

        result = detect_defects(image)
        output = result["annotated_image"]

        return av.VideoFrame.from_ndarray(output, format="bgr24")
