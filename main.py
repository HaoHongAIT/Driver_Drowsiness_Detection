import cv2
from flask import Flask, render_template, Response, make_response, request
from drowsiness_detection import drowsiness
from webcam import Webcam
from firebase import connect_firebase
from NpEncoder import NpEncoder
from time import time
import json

app = Flask(__name__)
score_list = [0]
data = []


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route("/start", methods=['GET', 'POST'])
def start():
    return render_template("demo.html")


def read_from_webcam():
    webcam = Webcam()
    drowsiness_model = drowsiness()
    while True:
        ret, image = next(webcam.get_frame())
        image = drowsiness_model.detect(ret, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        score_list.append(drowsiness_model.score)
        yield b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n--frame\r\n'


@app.route("/image_feed")
def image_feed():
    return Response(read_from_webcam(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/summary", methods=['GET', 'POST'])
def board():
    return render_template("board.html")


@app.route('/data', methods=["GET", "POST"])
def data():
    db = connect_firebase()
    score = score_list[-1]
    avg_score = sum(score_list) / len(score_list)
    data = [time() * 1000, score, avg_score]
    db.child("data").push(data)
    response = make_response(json.dumps(data, cls=NpEncoder))
    response.content_type = 'application/json'
    return response


@app.route('/test', methods=['GET', 'POST'])
def test():
    db = connect_firebase()
    if request.method == 'POST':
        if request.form['submit'] == 'add':
            name = request.form['name']
            db.child("todo").push(name)
            todo = db.child("todo").get()
            to = todo.val()
            return render_template('test.html', t=to.values())
        elif request.form['submit'] == 'delete':
            db.child("Driver_data").remove()
            db.child("todo").remove()
        return render_template('test.html')
    return render_template('test.html')


if __name__ == "__main__":
    app.run(debug=True)
