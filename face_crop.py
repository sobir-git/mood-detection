from cv2 import cv2


class FaceNotFoundError(Exception):
    pass


class FaceCropper:
    def __init__(self):
        casc_path = "resources/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(casc_path)

    def crop_face(self, image):
        """Take RGB image, crop the face and return."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        #
        # # Draw a rectangle around the faces
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        if len(faces) == 0:
            raise FaceNotFoundError()
        x, y, w, h = faces[0]  # gets the first face
        cropped = image[y:y + h, x:x + w, :]
        return cropped
