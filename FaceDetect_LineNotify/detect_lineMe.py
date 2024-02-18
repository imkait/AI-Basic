import sys
import time
import requests
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

def lineme():
  url = 'https://notify-api.line.me/api/notify'
  token = '9g6SAnsGGLWX8YrS1HuNOmxvQL3G9IsurvcnTvBy5r'
  headers = {'Authorization': 'Bearer ' + token}
  data = {'message':'偵測異常通知'}
  image=open('line_notify.jpg', 'rb')
  imageFile={'imageFile': image}
  r=requests.post(url, headers=headers, data=data, files=imageFile)
  print(r.status_code)

def run(model: str, min_detection_confidence: float,
        min_suppression_threshold: float, camera_id: int, width: int,
        height: int) -> None:

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 50  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 0)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  def save_result(result: vision.FaceDetectorResult, unused_output_image: mp.Image,
                  timestamp_ms: int):
      global FPS, COUNTER, START_TIME, DETECTION_RESULT

      # Calculate the FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()

      DETECTION_RESULT = result
      COUNTER += 1

  # Initialize the face detection model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.FaceDetectorOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.LIVE_STREAM,
                                       min_detection_confidence=min_detection_confidence,
                                       min_suppression_threshold=min_suppression_threshold,
                                       result_callback=save_result)
  detector = vision.FaceDetector.create_from_options(options)


  line_state=False
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run face detection using the model.
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    if DETECTION_RESULT:
        # print(DETECTION_RESULT)
        current_frame = visualize(current_frame, DETECTION_RESULT)
        
        if line_state==False:
           notice_time=time.time()
           cv2.imwrite('line_notify.jpg',current_frame)
           lineme()
           line_state=True
        else:
           if time.time()-notice_time>10:
              line_state=False


    cv2.imshow('face_detection', current_frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break

  detector.close()
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
    run('detect_face.tflite', 0.7 , 0.5 , 0 , 1280, 720)