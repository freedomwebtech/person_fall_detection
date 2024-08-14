import cv2    
import time
cpt = 0
maxFrames = 1000 # if you want 5 frames only.

count=0
cap=cv2.VideoCapture('fall5.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(1020,600))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\Users\freed\Downloads\yolov10-people-counting-main\yolov10-people-counting-main\images\img_%d.jpg" %cpt, frame)
    time.sleep(0.01)
    cpt += 1
    if cv2.waitKey(5)&0xFF==27:
        break
cap.release()   
cv2.destroyAllWindows()