from flask import Flask, jsonify, render_template, Response, request
import datetime
import time
from app.FaceRecognition import gen_frames
from app.Face_Detect import add_frames

data = {}           
file = []   

def index():
    return render_template('index.html')

def process():
        data['value'] = request.json['value']
        data['face_id'] = request.json['face_id']
        file.append(data)
        # process the data using Python code
        file.append(data)
        return data['value']
def register():
    return render_template('daftar.html')

def video_dataset():
    time.sleep(6)
    print(file)
    return Response(add_frames(file[0].get("value"), file[0].get("face_id")), mimetype='multipart/x-mixed-replace; boundary=frame')

def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')