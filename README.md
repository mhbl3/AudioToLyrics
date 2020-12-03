# audioToLyrics
 Deep Learning project for the generation of lyrics corresponding to given audio music genre.

## Data Generation
Attention: Make sure that any code is ran from ./audioToLyrics directory
To get started head to ./audiotolyrics/preprocessing/dataset_creation.ipynb
From this file the cells ran and the data set created. Note that a spotify, and a genius accounts are required to access for this step to work.
When running Spotipy, the user will be redirected by Spotify to a webpage to confirm access. It came to our attention that Google Colabs does not allow the web page to open. Therefore it is advised that the notebook is ran on a local machine. 

## Training Text Generator and Generating Text
Attention: Make sure that any code is ran from ./audioToLyrics directory
Head to ./audiotolyrics/textgen/example for an example on how to run the API developped. The cells are setup such that everything should run without a need to make changes. 
The user can make some changes if he/she wishes to explore some of the options. The notebook starts by creating "song" a container for the audio waves and log mel spectrograms. 
Another container is used for the text. With theese two, the model can be trained.
Towards the end of the notebook lyrics are generator for both the trained set and test set. The lyrics generated for the test set are exported to be used by the music genre classifier to access their quality ("how hip-hop or country" is the generated lyrics). 
