from flask import Blueprint, request, jsonify, Flask
import torch
import numpy as np
import db
import config
import pickle as pkl
from trainedModels.atlmodel import audioFeatureExtractor
from datacontainer import audioContainer, textContainer

torch.set_default_dtype(torch.float)
app = Flask(__name__)
api = Blueprint('api', __name__)

# class MyCustomUnpickler(pkl.Unpickler):
#     def find_class(self, module, name):
#         if module == "__main__":
#             module = "trainedModels.atlmodel"
#         return super().find_class(module, name)

# with open('out.pkl', 'rb') as f:
#     unpickler = MyCustomUnpickler(f)
#     obj = unpickler.load()

# required since audioContainer/textContainer were not originally saved in correct directory
class MyCustomUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "datacontainer"
        if "audiotolyrics" in module:
            module = "datacontainer"
        return super().find_class(module, name)

# Load pytorch model for inference
model_name = 'cpu_model.pt'
model_path = f'./trainedModels/{model_name}'


# with open(model_path, "rb") as f:
#     model = torch.load(f)

# model.eval()
print("Model loaded!")


@api.route('/generate', methods=['POST'])
def generate_lyrics():
    '''
    Endpoint to generate the new lyrics
    '''
    if request.method == 'POST':
        data = request.json
        if 'audio_input' not in data:
            return jsonify({'error': 'No audio uploaded'}), 400
        else:
            with open("./textcontainer.pkl", "rb") as f:
                unpickler = MyCustomUnpickler(f)
                training_text = unpickler.load()
                print("tokenizer loaded")
            with open(model_path, "rb") as f:
                model = torch.load(f)
                model.float()
            tokenizer = training_text.tokenizer
            audio_input = torch.from_numpy(np.array(data['audio_input'])).type(torch.float)
            # audio_input = ast.literal_eval(audio_input)
            text_input = data["text_input"]
            num_lines = data["num_lines"]
            lyrics = ""
            for _ in range(int(num_lines)):
                gen_lyrics = model.spitbars(tokenizer, audio_input,
                                        cuda=False, in_text=text_input,
                                        use_spectrogram=True)
                gen_lyrics = gen_lyrics.replace("startseq", "").replace("endseq","")#split("startseq")[1 if "startseq" in gen_lyrics else 0].split("endseq")[0]
                lyrics = lyrics + gen_lyrics + "\n"

            return jsonify(lyrics)

@api.route('/save', methods=['POST'])
def post_lyrics():
    '''
    Save lyrics to database
    '''
    if request.method == 'POST':
        expected_fields = [
            'filename',
            'gen_lyrics',
            "time_song",
            "start_word",
        ]

        if any(field not in request.form for field in expected_fields):
            return jsonify({'error': 'Missing field in body'}), 400

        query = db.GenLyrics.create(**request.form)

        return jsonify(query.serialize())

@api.route('/load', methods=['GET'])
def get_lyrics():
    '''
    Get all lyrics.
    '''
    if request.method == 'GET':
        query = db.GenLyrics.select().order_by(db.GenLyrics.created_date.desc())
        return jsonify([r.serialize() for r in query])

app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST)