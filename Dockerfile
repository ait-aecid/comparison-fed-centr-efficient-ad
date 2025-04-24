FROM python:3.12

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U "flwr[simulation]"

COPY . .

CMD ["./run_multiple_runs.sh"]