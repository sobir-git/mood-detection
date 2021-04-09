# Mood detection

It identifies your mood level. Negative, positive, or zero.


### Description:
1. Uses opencv library to detect and crop the face.
2. Uses a modified MobileNetV2 for mood regression. (mood level is any number between -1 and 1)

### Model
Model training is in the [notebook](facial_expression_mood_regression.ipynb).
1. The model is initially trained on [FER dataset](https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge).
2. Then modifying the head, it is trained (transfer learning) to my mood regression personal dataset.
