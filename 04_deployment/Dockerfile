FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

RUN mkdir outputs

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["python", "predict.py", "--year", "2022", "--month", "4", "--model", "model.bin"]