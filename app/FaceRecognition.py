import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

    
def gen_frames():
    dataset_folder = "dataset/"
    name = ""
    face_id = ""
    names = []
    images = []
    for folder in os.listdir(dataset_folder):
        for name in os.listdir(os.path.join(dataset_folder, folder))[:100]: # limit only 70 face per class
            if name.find(".jpg") > -1 :
                img = cv2.imread(os.path.join(dataset_folder + folder, name))
                images.append(img)
                names.append(folder)
                
    labels = np.unique(names)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


    def detect_face(img, idx):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        try :
            x, y, w, h = faces[0]

            img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (100, 100))
        except :
            img = None
        return img

    croped_images = []
    for i, img in enumerate(images) :
        img = detect_face(img, i)
        if img is not None :
            croped_images.append(img)
        else :
            del names[i]

    for label in labels:
        ids = np.where(label== np.array(names))[0]
        images_class = croped_images[ids[0] : ids[-1] + 1] # select croped images for each class
        
    name_vec = np.array([np.where(name == labels)[0][0] for name in names])
    model = cv2.face.LBPHFaceRecognizer_create()

    # model.train(croped_images, name_vec)

    # model.save("lbph_model.yml")
    model.read("lbph_model.yml")

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
    URL = "http://192.168.61.115:81/stream"
    cap = cv2.VideoCapture(0)
    confidence = 0
    while cap.isOpened() :
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                
                idx, confidence = model.predict(face_img)
                if (confidence < 100 ):
    
                    label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                    name = labels[idx].replace("_", " ").split("-")[0]
                    face_id = labels[idx].split("-")[1]
       
                else:
                    label_text = "%s (%.2f %%)" % ("unknown", confidence)
                frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))

        else :
            break
        if cv2.waitKey(10) == ord('q'):
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



                        
    cv2.destroyAllWindows()
    cap.release()
    
gen_frames()