import os
import cv2
import time

import torch

from pipeline import NNPredictor
from face_crop import FaceCropper
from utils import MoodModel
from warning import draw, red, green

video_capture = cv2.VideoCapture(0)
fc = FaceCropper()

moodmodel = MoodModel()
moodmodel.load_state_dict(torch.load('resources/moodmodel.pt', map_location='cpu'))
nn = NNPredictor(moodmodel)

# shoot_mode = 'negative'
# shoot_mode = 'positive'
shoot_mode = None

savedir = f'new_data/{shoot_mode}'
os.makedirs(savedir, exist_ok=True)

every = 2  # capture every seconds

prediction = 0
capture_time = time.time()


def save_frame(frame):
    # cv2.imshow('Video', frame)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f'{savedir}/WIN_{timestr}.JPG', frame)


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        time.sleep(5)
        pass

    # time.sleep(max(0, every - time.time() + capture_time))

    size = 12
    if prediction is not None:
        if prediction < -0.3:
            draw(red, size)
        elif prediction > 0.3:
            draw(green, size)

    time.sleep(0.01)
    if time.time() - capture_time < every:
        continue

    capture_time = time.time()
    ret, frame = video_capture.read()

    # downsample
    frame = frame[::4, ::4, :]

    # show video
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        # print('will predict ...')
        break
    start = time.time()
    prediction = nn.compute_mood([frame])[0]
    if prediction is None:
        print('face not detected')
        continue
    print(f'mood: {prediction:.1f}  ({time.time() - start:.3f}s)  ({time.time() - capture_time:.3f}s)')

    if (shoot_mode == 'positive' and prediction < -0.5) \
            or (shoot_mode == 'negative' and prediction > 0.5):
        save_frame(frame)
        print('saved frame')

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
