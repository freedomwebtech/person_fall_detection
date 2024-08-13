import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cvzone



model = YOLO("best.pt")  

cap=cv2.VideoCapture('fall.avi')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0
while True:
    ret,frame = cap.read()
    
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    # Resize frame
    frame = cv2.resize(frame, (1020, 600))
    
    # Perform segmentation
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        print(c)
        
        if 'person' in c:
              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
              cvzone.putTextRect(frame,f'{c}',(x2,y2),1,1)
        else:
            'person_fall' in c
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)      
            
    # Overlay segmentation mask on the original frame
    for result in results:
        if result.masks is not None:
           for j, mask in enumerate(result.masks.data):
               mask = mask.numpy() * 255
               mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
               mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
               contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               cv2.fillPoly(mask_bgr, contours, (0, 0, 255))
               frame = cv2.addWeighted(frame, 1, mask_bgr, 0.5, 0)
    
    # Display frame
    cv2.imshow("FRAME", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
