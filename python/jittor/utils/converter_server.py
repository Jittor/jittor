from flask import Flask
from flask import request
from flask import jsonify
app = Flask(__name__)
import json

from jittor.utils.pytorch_converter import convert

@app.route('/', methods=["GET", "POST"])
def hello():
    msg = request
    data = msg.data.decode("utf-8") 
    try:
        data = json.loads(data)
        src = data["src"]
        pjmap = json.loads(data["pjmap"])
        jt_src = convert(src, pjmap)
    except Exception as e:
        jt_src = str(e)
    response = jsonify(jt_src=jt_src)

    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0")