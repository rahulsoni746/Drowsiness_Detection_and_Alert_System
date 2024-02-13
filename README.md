
# Drowsiness-Detection-&-Alert-System-Using-Deep-Learning

This Deep Learning project detect the drowsiness and sleepiness and alert the human being to meet an accident.

We use latest deep learning model which provide the accuracy approximately than 70%-80%. We use YOLOv8 model which can detect multiple features of human face.

## Dependencies
•	Ultralitics

•	Torch

•	labelimg

•	time

•	numpy

•	pyttsx3

## Procedure

#Open terminal and run **_{```pip install requirements.txt```}_**

#And run **"run.py"**

## Training

Use labelimg library to label the images to create a label data set and train our model to 

<img src="Source\Images\Labeling.png" alt="Input Image" style="width:800px;"/>

### Traning Data
<img src="Source\Images\TraningData.png" alt="Output Image" style="width:800px;"/>

## Traning Rusult

<img src="Source\Images\results.png" alt="Output Image" style="width:800px;"/>

## Testing
1.	After that we provide the camera access or input image to model.

2.	Model will extract the feature from the model and perform conditional  operation on the given feature.

3.	After that it will either alert or continue the execution till any interruption.

## Code

```
import cv2
import torch
from ultralytics import YOLO

# give the path of model 
model = YOLO("sample_model\model.onnx")

# if you want to predict on custom date (photos or videos) just replace zero "0" with path of data
cap = cv2.VideoCapture(0)

while cap.isOpened():

    _,frame = cap.read()
    
    results = model.predict(source=frame,show=True,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    #press ESC or 'q' or 'Q' for exit
    if cv2.waitKey(2) & 0xFF == 27 or ord('q') or ord('Q'):
        
        break
    
cap.release()

cv2.destroyAllWindows()

```

## Input Image
<img src="Source\Images\input.jpg" alt="Input Image" style="width:600px;"/>

## Predicted Image
<img src="Source\Images\output.jpg" alt="Output Image" style="width:600px;"/>

