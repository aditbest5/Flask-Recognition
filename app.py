from flask import Flask, jsonify, render_template, Response, request
import datetime
import time
import FaceRecognition
import Face_Detect


URL = "http://192.168.51.115:81/stream"
data = {}           
file = []   

app=Flask(__name__)






@app.route('/')
def index():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():
        data['value'] = request.json['value']
        data['face_id'] = request.json['face_id']
        file.append(data)
        # process the data using Python code
        file.append(data)
        return data['value']
@app.route('/daftar')
def register():
    return render_template('daftar.html')
@app.route('/video_dataset')
def video_dataset():
    time.sleep(6)
    print(file)
    return Response(Face_Detect.add_frames(file[0].get("value"), file[0].get("face_id")), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed')
def video_feed():
    return Response(FaceRecognition.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)