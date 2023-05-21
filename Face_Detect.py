import cv2
import os
import time


def add_frames(name, faceId):
    URL = "http://192.168.61.115:81/stream"

    time.sleep(3)
    cap = cv2.VideoCapture(0)
    dataset_folder="dataset/"
    my_name = name.replace(" ", "_")
    face_id = faceId
    folder_name = my_name + "-" + face_id
    os.mkdir(dataset_folder + folder_name)
    num_sample = 100
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    i = 0
    def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):
            
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img,
                    (x0, y0 + baseline),  
                    (max(xt, x0 + w), yt), 
                    color, 
                    2)
        cv2.rectangle(img,
                    (x0, y0 - h),  
                    (x0 + w, y0 + baseline), 
                    color, 
                    -1)  
        cv2.putText(img, 
                    label, 
                    (x0, y0),                   
                    cv2.FONT_HERSHEY_SIMPLEX,     
                    0.5,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if ret :
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                cv2.imshow("Capture Photo", frame)
                cv2.imwrite("dataset/%s/%s_%04d.jpg" %  (folder_name, folder_name, i), cv2.resize(frame, (250,250)))
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(100) == ord('q') or i == num_sample:
                break
            i += 1    
    cap.release()
    cv2.destroyAllWindows()