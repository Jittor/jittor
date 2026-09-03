"""HTTP front end for the PyTorch-source converter.

Lives next to :mod:`jittor.compat.pytorch_converter`, the translator it
exposes, rather than in the former ``jittor/utils`` drawer. Its launcher is
``tools/services/legacy/converter_server.sh``; that script installs the
published wheel inside a container and names this module in ``FLASK_APP``,
so the two have to agree.
"""

from flask import Flask
from flask import request
from flask import jsonify
app = Flask(__name__)
import json

from jittor.compat.pytorch_converter import convert

@app.route('/', methods=["GET", "POST"])
def hello():
    msg = request
    data = msg.data.decode("utf-8") 
    try:
        data = json.loads(data)
        src = data["src"]
        pjmap = json.loads(data["pjmap"])
        jt_src = convert(src, pjmap)
    except (SyntaxError, ValueError, KeyError, TypeError, AttributeError,
            IndexError, NotImplementedError) as exc:
        # This endpoint's product is the translation *or* the reason there is
        # not one, so the message is the response rather than a swallowed
        # failure. The list is what the request path can actually raise:
        # SyntaxError and ValueError (which json.JSONDecodeError and
        # UnicodeDecodeError subclass) from parsing the posted source and its
        # pjmap, KeyError from a request with no "src", and the
        # TypeError/AttributeError/IndexError/NotImplementedError the
        # translator raises on a construct it has no mapping for. A
        # MemoryError or a RecursionError is not a translation result and
        # must reach the client as a failed request.
        jt_src = str(exc)
    response = jsonify(jt_src=jt_src)

    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0")