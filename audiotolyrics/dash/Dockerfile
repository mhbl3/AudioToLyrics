FROM python:3.7

ADD dash/requirements.txt /dash/app/

COPY dash /dash/app/
COPY preprocessing/datacontainer.py /dash/app/
COPY trained_model/textcontainer.pkl /dash/app/

WORKDIR /dash/app

RUN pip install SoundFile
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1
RUN apt-get install -y ffmpeg

RUN pip install -r requirements.txt
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:app.server"]