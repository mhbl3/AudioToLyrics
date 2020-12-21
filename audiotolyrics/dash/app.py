import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash
import numpy as np
import base64
import os
import pickle as pkl
import requests
import config
import pandas as pd
from flask import request
from datacontainer import audioContainer, textContainer
# from audiotolyrics.textgenmodel.atlmodel import audioFeatureExtractor


class MyCustomUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "datacontainer"
        if "audiotolyrics" in module:
            module = "datacontainer"
        return super().find_class(module, name)


# with open('out.pkl', 'rb') as f:
#     unpickler = MyCustomUnpickler(f)
#     obj = unpickler.load()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,
                # external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
# Might need to change this
TOTAL_TIME = 60*5 # 5 minutes
song_slider_dict = {}
for i, time_sec in enumerate(np.arange(0, TOTAL_TIME+30, 60)):
    song_slider_dict.update({int(time_sec):f"{i} min"})

file_upload = \
    html.Div([dcc.Upload(
        id ="upload-data",
        children=[
            html.Div([
                "Drag and Drop or ",
                html.A("Select file")
            ],style={"margin-top":"12px"})

        ],
        style={
            "width":"20%",
            "height":"100px",
            "lineHeight":"60px",
            "borderWidth":"1px",
            "borderStyle":"dashed",
            "borderRadius":"5px",
            "textAlign":"center",
            "margin":"10px"
        },
        multiple=False,

    )]
    )
div_file_upload = html.Div(id="output-data-upload",
            # children=[html.Div("File uploaded name")],
             style={"word-wrap": "break-word",
                    "margin-top":"20px",
                    "width":"300px"})

song_slider = dcc.Slider(
    id="song-slider",
    min=0,
    max= TOTAL_TIME,
    step= 1,
    value=30,
    marks= song_slider_dict,
    included=False
)

slider_html = html.Div(id="slider-container",
                      children = [song_slider,
                                  html.Div(id='output-container-range-slider')],
                       style={"width":"900px",
                              "margin-left":"315px",
                              "margin-top":"-120px"})
start_word = dbc.Textarea(id="start-word",
                          placeholder="Enter starting word/sentence",
                          style={"resize":"none",
                                 "width":"400px"})

html_start_word = html.Div([start_word, html.Div(id="div start-word")],
                           style={"margin-left":"315px"})

# alert = dbc.Spinner(
#                     html.Div(id="alert-message")
#                   )

n_lines = dbc.Input(id="number-lines",
                    placeholder="Number of lines",
                    min=1,
                    max=30,
                    style={"width":"100px"})
html_n_lines = html.Div(id="div number-lines",
                        children=[n_lines],
                        style={"margin-left":"315px"})

text_box = dcc.Textarea(
            id="text-box",
            placeholder="Lyrics appear here",
            readOnly=True,
            draggable=False,
            contentEditable = False,
            style={"margin-left":"500px",
                   "height":"500px",
                   "margin-top":"10px",
                   "width":"500px",
                   "resize":"none"})

app.title = "ATL"
app.layout = html.Div([dcc.Location(id="url", refresh=False),
                       html.Div(id="page-content",
                                children=[html.P(id="placeholder")])])
saved_lyrics_page = html.Div(
    [ html.H1("ATL: Saved Lyrics"),
        html.Div(id="lyrics-page-content"),
        html.P(
            dcc.Link("Go to Home Page", href="/"),
            style={"marginTop": "20px"}
        )
    ]
)
home_layout = dbc.Container(
    id="master-container",
    children= [
        html.H1("ATL: Audio To Lyrics"),
        html.Hr(),
        dbc.Row([
            dbc.Col([file_upload]),
            dbc.Col([slider_html])
        ]),
        dbc.Row([
            dbc.Col(html_start_word),
            dbc.Col(html_n_lines)]),
        dbc.Row([dbc.Col(div_file_upload)]),
        html.Hr(),
        dbc.Row([dbc.Col([dbc.Button("Refresh",
                                     id="my-button",
                                     n_clicks= 0,
                                     style={"margin-left": "10px"}
                                     )]),
                 dbc.Col([dbc.Button("Submit Lyrics",
                                     id="submit my-button",
                                     type="submit",
                                     style={"margin-left": "10px",
                                            "margin-top": "10px"}
                                     )])
                 ]),
        dbc.Row([dbc.Col([html.Div(id="submitted_text",
                                   style={"margin-left": "10px"})])]),
        # html.Hr(style={"margin-top":"20px"}),
        dbc.Row([dbc.Card(text_box)]),
        dbc.Row([dbc.Col([dcc.Link("Go to Saved Lyrics Page", href="/saved-lyrics")])])
    ],
    style={"margin":"auto"}
)


def parse_contents_audio(contents, filetype):
    content_type, content_string = contents.split(',')
    rawdata = base64.b64decode(content_string)

    if filetype == 'mp3':
      fh = open("./tmp.mp3", "wb")
      fh.write(rawdata)
      fh.close()

    elif filetype =='wav':
      fh = open("./tmp.wav", "wb")
      fh.write(rawdata)
      fh.close()

    with open("./tmp.txt", "w") as f:
        f.write("song nothing important\nReally not important")


@app.callback(
    Output("submit my-button", "disabled"),
    Input("text-box", "value")
)
def change_submit(value):
    if value is None:
        return True
    else:
        return False

@app.callback(
    Output("submitted_text","children"),
    Input("submit my-button", "n_clicks"),
    [State("text-box","value"),
    State("upload-data", "filename"),
    State("song-slider", "value"),
    State("start-word", "value")
     ]
)
def send_data_to_db(n_click, gen_lyrics, filename,
                    time_song, start_word):
    if n_click is not None:
        if start_word is None:
            start_word = "startseq"
        response = requests.post(
            f"{config.API_URL}/save",
            data={
                'gen_lyrics': gen_lyrics,
                'filename': filename,
                'time_song':time_song,
                'start_word':start_word,
            }
        )

        if response.ok:
            print("Lyrics Saved")
        else:
            print("Error Saving Lyrics")
        return "Thanks for submitting!"

@app.callback(
    Output('lyrics-page-content', 'children'),
    [Input("url", "pathname")]
)
def load_lyrics_table(pathname):# filename, time_song, start_word):
    if pathname != "/saved-lyrics":
        return None

    response = requests.get(f"{config.API_URL}/load")
    gen_lyrics = pd.DataFrame(response.json())

    table = dbc.Table.from_dataframe(gen_lyrics,
                                     striped=True,
                                     bordered=True,
                                     hover=True,
                                     responsive=True,
                                     header=["Date","File name", "Sampled time", "Generated lyrics starting word/sentence",
                                             "Generated Lyrics"],
                                     columns=["created_date","filename", "time_song", "start_word",
                                             "gen_lyrics"]
                                     )
    return table

@app.callback(
    Output('output-container-range-slider', "children"),
    [Input("song-slider", "value")]
)
def update_slider(value):
    return f"Audio starts at {value} seconds"

@app.callback(
    [Output("output-data-upload", "children"),
    Output("text-box","value")],
    [Input("upload-data","contents"),
    Input("my-button","n_clicks")],
    [State("upload-data", "filename"),
    State("song-slider", "value"),
    State("start-word", "value"),
    State("number-lines", "value")]
)
def parse_input(content, n_clicks, filename, value, first_word_sentence, num_lines):
    if content is not None:
        if num_lines is None:
            num_lines = 1
        if ("mp3" not in filename ) and ("wav" not in filename):
            raise Exception("Insert audio data (.wav or .mp3)")
        else:
            if "mp3" in filename:
                filetype = "mp3"
                # response = requests.post(
                #     f"{config.API_URL}/process", data={'contents': content,
                #                                        'fileType':filetype})
                parse_contents_audio(content, filetype)
            else:
                filetype = "wav"
                # response = requests.post(
                #     f"{config.API_URL}/predict", data={'contents': content,
                #                                        'fileType': filetype})
                parse_contents_audio(content, filetype)
            songs = audioContainer([os.getcwd()], sr=16000, offset=value, song_duration=10,
                                   use_log_spectrogram=True)
            songs.load_songs()
            text = textContainer([os.getcwd()], songs, text_limit_per_song=1)
            _, audio_input = text.run_datapipeline()

            if first_word_sentence is None:
                word = "startseq"
            else:
                word = first_word_sentence

            the_key = list(audio_input.keys())[0]
            audio_input = audio_input[the_key][None, :, :]
            json_out = {'audio_input':audio_input.tolist(),
                        'text_input':word,
                        'num_lines':num_lines}
            response = requests.post(
                f"{config.API_URL}/generate", json=json_out)
            lyrics = response.json()
            os.remove("./tmp.mp3")
            os.remove("./tmp.txt")
            return f"File uploaded: {filename}", lyrics
    else:
        try:
            os.remove("./tmp.mp3")
            os.remove("./tmp.txt")
        except:
            pass
        return None, None

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def page_display(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == "/saved-lyrics":
        return saved_lyrics_page

if __name__ == '__main__':
    app.run_server(debug=True)