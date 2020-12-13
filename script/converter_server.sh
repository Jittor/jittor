cat > /tmp/converter_server.dockerfile <<\EOF
FROM jittor/jittor

RUN python3.7 -m pip install flask
RUN apt update && apt install git -y
EOF

sudo docker build --tag jittor/converter_server -f /tmp/converter_server.dockerfile .
sudo docker run --rm jittor/converter_server bash -c "python3.7 -m pip install -U git+https://github.com/Jittor/jittor.git && python3.7 -m jittor.utils.converter_server"