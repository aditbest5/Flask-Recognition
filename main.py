from flask import Flask
from app import views

app = Flask(__name__) # webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='face-recognition',view_func=views.index)
app.add_url_rule(rule='/process/',endpoint='process',view_func=views.process, methods=['POST'])
app.add_url_rule(rule='/daftar/',endpoint='register',view_func=views.register)
app.add_url_rule(rule='/video_dataset/',endpoint='video_dataset',view_func=views.video_dataset)
app.add_url_rule(rule='/video_feed/',endpoint='video_feed',view_func=views.video_feed)

if __name__ == "__main__":
    app.run(debug=True)