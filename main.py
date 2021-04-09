import cv2
import time
import torch
from pipeline import NNPredictor
from face_crop import FaceCropper
from utils import MoodModel
from warning import draw, red, green, black

# capture every n seconds
EVERY = 1
fc = FaceCropper()

# initialize mood model
moodmodel = MoodModel()
moodmodel.load_state_dict(torch.load('resources/moodmodel.pt', map_location='cpu'))
nn = NNPredictor(moodmodel)


class MoodSM:
    """Mood state machine."""

    def __init__(self, momentum=0.5):
        self._mood = 0
        self.momentum = momentum

    def update(self, mood_prediction):
        self._mood = self._mood * self.momentum + mood_prediction * (1 - self.momentum)
        return self._mood

    def get_mood(self):
        return self._mood


def main():
    # initialize mood state machine
    mood_sm = MoodSM(momentum=0.618)

    # webcam
    video_capture = cv2.VideoCapture(0)

    capture_time = time.time()

    # current frame color
    color = None

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            time.sleep(5)
            pass

        time.sleep(0.1)
        if time.time() - capture_time < EVERY:
            continue

        # maybe quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        capture_time = time.time()
        _, frame = video_capture.read()

        # downsample
        frame = frame[::4, ::4, :]

        start = time.time()
        prediction = nn.compute_mood([frame])[0]
        if prediction is None:
            print('face not detected')
            continue

        # update mood
        mood = mood_sm.update(prediction)

        size = 12
        if mood < -0.3:
            new_color = red
        elif mood > 0.3:
            new_color = green
        else:
            new_color = black

        # color transition logic
        update = True
        if new_color == color:  # if same color -> do not update
            update = False
        if new_color is black and color is green:  # never green -> black
            update = False
        if new_color is red:
            update = True

        if update:
            draw(new_color, size)
            color = new_color

        print(f'mood: ({prediction:.1f}, {mood_sm.get_mood():.1f})  '
              f'({time.time() - start:.3f}s)  ({time.time() - capture_time:.3f}s)')

    # release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()