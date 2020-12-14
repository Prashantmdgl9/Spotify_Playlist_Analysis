#Imports

import pandas as pd
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import turicreate as tc
from math import pi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
%matplotlib inline


pl_id = #Use your own playlist id

client_id = # Use your own client id
client_secret =  # Use your own client secret
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

results = sp.playlist_items(pl_id)

#Prints entre JSON object
print(json.dumps(results, indent = 4))



#Extract tracks Ids and track names from the playlist

offset = 0
track_ids = []
track_names = []
while offset < 1500:
    response_i = []
    response_n = []
    response_n.append(sp.playlist_items(pl_id, fields = 'items.track.name' , offset = offset))
    response_i.append(sp.playlist_items(pl_id, fields = 'items.track.id' , offset = offset))
    offset = offset + 100
    cnt = 0
    for d in response_n[0]['items']:
        val = response_n[0]['items'][cnt]['track']['name']
        if val != 'None':
            track_names.append(val)
        cnt = cnt + 1
    cnt_i = 0
    for d in response_i[0]['items']:
        val = response_i[0]['items'][cnt_i]['track']['id']
        if val != 'None':
            track_ids.append(val)
        cnt_i = cnt_i + 1

len(track_ids)
len(track_names)

#Create a dataframe
track_recs = {'Track': track_ids, 'Name' : track_names}
df_interim = pd.DataFrame(track_recs, columns = ['Track', 'Name'])
df_interim.shape
#Remove any null values
df_interim = df_interim.mask(df_interim.eq('None')).dropna()
df_interim.shape

df_interim
##############

#Extract the quantitative features for the tracks

tracks = {}
tracks['acousticness'] = []
tracks['danceability'] = []
tracks['energy'] = []
tracks['instrumentalness'] = []
tracks['liveness'] = []
tracks['loudness'] = []
tracks['speechiness'] = []
tracks['tempo'] = []
tracks['valence'] = []
tracks['uri'] = []
tracks['duration'] = []

for track in df_interim['Track']:
    features = sp.audio_features(track)
    tracks['acousticness'].append(features[0]['acousticness'])
    tracks['danceability'].append(features[0]['danceability'])
    tracks['energy'].append(features[0]['energy'])
    tracks['instrumentalness'].append(features[0]['instrumentalness'])
    tracks['liveness'].append(features[0]['liveness'])
    tracks['loudness'].append(features[0]['loudness'])
    tracks['speechiness'].append(features[0]['speechiness'])
    tracks['tempo'].append(features[0]['tempo'])
    tracks['valence'].append(features[0]['valence'])
    #tracks['popularity'].append(features[0]['popularity'])
    tracks['uri'].append(features[0]['uri'])
    tracks['duration'].append(features[0]['duration_ms'])


dataframe = pd.DataFrame.from_dict(tracks)
dataframe
dataframe['name'] = df_interim['Name']

dataframe
dataframe.to_csv('Sound/Data/Mosaic.csv')

# Read the file
dataframe = pd.read_csv('Sound/Data/Mosaic.csv')


# Embeddings and visualisation
df = dataframe.drop(['uri', 'name', 'Unnamed: 0'], axis = 1)
df
df.describe()

dataframe[dataframe['name'] == 'Stockholm Skies (feat. Tom Taped & Alex Aris)']
df_s = StandardScaler().fit_transform(df)
df_s

tsne_results = TSNE(n_components = 2, verbose = 1, perplexity = 50, n_iter = 10000).fit_transform(df_s)

tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
tsne_results['name'] = dataframe['name']
tsne_results['energy'] = dataframe['energy']
tsne_results['acousticness'] = dataframe['acousticness']
tsne_results['valence'] = dataframe['valence']
tsne_results['tempo'] = dataframe['tempo']
tsne_results['danceability'] = dataframe['danceability']
tsne_results['uri'] = dataframe['uri']

tsne_results.head(5)
tsne_results[tsne_results['name'] == 'Stockholm Skies (feat. Tom Taped & Alex Aris)']

def plot_tsne():
    plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'])
    plt.show()


plot_tsne()

def plt_tsne():
    sns.FacetGrid(tsne_results, hue = 'danceability', height = 5).map(plt.scatter, 'tsne1', 'tsne2' )
    plt.show()

plt_tsne()

def plot_tsne():
    px.scatter(tsne_results, x="tsne1", y="tsne2",color="energy", hover_data=["name"]).write_html('Sound/Data/tsne_mosaic.html', auto_open = True)



def plot_tsne():
    fig = px.scatter(tsne_results, x="tsne1", y="tsne2",color="energy", hover_data=["name"])
    fig.show()


plot_tsne()

fig = go.Figure(data=go.Scatter(x=tsne_results['tsne1'],
    y =tsne_results['tsne2'],
    mode='markers',
    marker=dict(
        color=tsne_results['acousticness'], #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )
))
fig.show()

tc.config.set_runtime_config('TURI_NUM_GPUS', 2)
tc.visualization.set_target(target='browser')
sf = tc.SFrame.read_csv('Sound/Data/Lena.csv', verbose = True)
sf.print_rows(5)
sf.show()



radar_n = ['acousticness', 'danceability', 'energy', 'valence', 'instrumentalness', 'liveness']
df_radar = df[radar_n]
df_radar

df_radar.mean()


r = [0.231858, 0.562153, 0.666474, 0.415243, 0.172336, 0.183510]
df_x = pd.DataFrame(dict(
    r=r,
    theta=radar_n))
df_x


x = df_x[['r']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
r_scaled = min_max_scaler.fit_transform(x)
r_scaled
lst = []
for val in r_scaled:
    lst.append(val[0])
lst

df_x = pd.DataFrame(list(zip(radar_n, lst)), columns =['attr', 'scale'])
df_x


N = len(radar_n)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
angles
values = lst
values += values[:1]
values
def pol_plot():
    fig = plt.figure(figsize = (10,10))
    ax = plt.subplot(polar=True)
    plt.polar(angles, values)
    plt.fill(angles, values, alpha = 0.3)
    plt.xticks(angles[:-1], radar_n)
    ax.set_rlabel_position(0)
    plt.yticks([0,0.4,0.6, 0.8], color = 'grey', size = 7)
    plt.show()
pol_plot()



tsne_results['index'] =  np.arange(len(tsne_results))
tsne_results
cols = ['tsne1', 'tsne2']
tsne_results_cos = tsne_results[cols ]
similarity = cosine_similarity(tsne_results_cos)

similarity.shape

idx = tsne_results[tsne_results['name'] == "Sansa"]
idx = idx['index'].values


df_temp = pd.DataFrame(list(zip(tsne_results['name'], similarity[idx[0]].tolist())), columns =['name', 'distance'])
df_temp['uri'] = tsne_results['uri']

df_temp.sort_values('distance', ascending = False)[0:15]
