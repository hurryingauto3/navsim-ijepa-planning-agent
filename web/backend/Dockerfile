FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install "fastapi[standard]" pydantic==2.* numpy==2.* python-socketio[asgi]==5.* redis==5.* boto3==1.* uvicorn==0.*

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
