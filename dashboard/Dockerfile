FROM python:3.11-slim

WORKDIR /app
COPY server.py index.html ./

ENV RUNS_ROOT=/data/runs \
    LAUNCH_LOG=/data/logs/kd_freq_3seed.log \
    PORT=8080

EXPOSE 8080
CMD ["python3", "server.py"]
