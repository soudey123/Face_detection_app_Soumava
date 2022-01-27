FROM python:3.8-slim-buster
LABEL maintainer = "Soumava Dey"
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "app.py"]