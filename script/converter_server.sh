cat > /tmp/converter_server.dockerfile <<\EOF
FROM jittor/jittor

RUN python3.7 -m pip install flask
RUN apt update && apt install git -y
EOF

docker build --tag jittor/converter_server -f /tmp/converter_server.dockerfile .

# docker run --rm -it -m 16g --cpus=8 -p 0.0.0.0:5000:5000  jittor/converter_server bash -c "python3.7 -m pip install -U git+https://github.com/Jittor/jittor.git && python3.7 -m jittor.utils.converter_server"
while true; do
    timeout --foreground 24h docker run --rm -it -m 16g --cpus=8 -p 0.0.0.0:5000:5000 -v ~/https:/https jittor/converter_server bash -c "python3.7 -m pip install -U git+https://github.com/Jittor/jittor.git && python3.7 -m jittor.test.test_core && FLASK_APP=/usr/local/lib/python3.7/dist-packages/jittor/utils/converter_server python3.7 -m flask run --cert=/https/fullchain.pem --key=/https/privkey.pem --host=0.0.0.0"
    sleep 10
done