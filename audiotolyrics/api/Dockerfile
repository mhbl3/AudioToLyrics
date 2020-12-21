FROM python:3.7

ADD api/requirements.txt /api/app/

COPY api /api/app/
COPY textgenmodel /api/app/
COPY preprocessing/datacontainer.py /api/app
COPY trained_model/textcontainer.pkl /api/app

WORKDIR /api/app

RUN pip install SoundFile
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt


EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]