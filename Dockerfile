FROM python:3.13-slim
WORKDIR /app
COPY . .
ENV AGENT_HOST=0.0.0.0 AGENT_PORT=8080 AGENT_DATA_DIR=/app/data
EXPOSE 8080
CMD ["python", "-m", "agent_native"]

