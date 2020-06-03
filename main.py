
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import json
import pandas as pd
import keras
import tensorflow as tf
from keras import backend
from keras.models import load_model
from tensorflow.python.keras.backend import set_session


sess = tf.Session(config=None)
graph = tf.get_default_graph()
set_session(sess)

model = load_model('collab_trained_smaller.h5')




app = Flask(__name__)



#loading some data
link = pd.read_csv('links_small.csv')[['movieId']]
link_tmdb = pd.read_csv('links_small.csv')[['tmdbId']]
#pre-processing the arrays
np_link = link.to_numpy()
flatten_np_link = np_link.flatten()
np_link_tmdb = link_tmdb.to_numpy()
flatten_np_link_tmdb = np_link_tmdb.flatten()

#removing index to not give input above a range
remove_index = np.argwhere(flatten_np_link >= 163949 )
removed_np_link = np.delete(flatten_np_link, remove_index)
removed_np_link_tmdb = np.delete(flatten_np_link_tmdb, remove_index)







@app.route('/prediciton', methods=["POST"])
def index():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        #getting the userId from front-end
        message = request.get_json(force=True)
        py_testuser = json.loads(message['user']) #makes a python variable of type string
        int_testuser = int(py_testuser) #converts string into int
        int_testuser = [int_testuser]#turn that variable into a python list
        np_testuser = np.array(int_testuser)#turn that list to numpy array

        # #pre-processing the arrays
        # np_link = link.to_numpy()
        # flatten_np_link = np_link.flatten()
        # np_link_tmdb = link_tmdb.to_numpy()
        # flatten_np_link_tmdb = np_link_tmdb.flatten()
        #
        # #removing index to not give input above a range
        # remove_index = np.argwhere(flatten_np_link >= 163949 )
        # removed_np_link = np.delete(flatten_np_link, remove_index)
        # removed_np_link_tmdb = np.delete(flatten_np_link_tmdb, remove_index)

        #creating a repeated user array
        array_size = removed_np_link.size
        np_testuser_repeated = np.repeat(np_testuser, array_size)




        #making predictions
        rating = model.predict([np_testuser_repeated, removed_np_link] , batch_size=2)


        # removing movies with rating below 4
        remove_index_rating = np.argwhere(rating < 4 )
        best_rating = np.delete(removed_np_link, remove_index_rating)
        best_rating_tmdb = np.delete(removed_np_link_tmdb, remove_index_rating)
        remove_index_nan = np.argwhere( np.isnan(best_rating_tmdb))

        final_rating = np.delete(best_rating, remove_index_nan)
        final_rating_tmdb = np.delete(best_rating_tmdb, remove_index_nan)
        final_rating_tmdb_unique = np.unique(final_rating_tmdb)


        if final_rating_tmdb_unique.size > 50:
           final_rating_tmdb_reduced = np.random.choice(final_rating_tmdb_unique, 50)
        else:
            final_rating_tmdb_reduced = final_rating_tmdb_unique


        #reversing the array so the newer movies are in the front
        reverse_tmdb = final_rating_tmdb_reduced[::-1]

        json_tmdb = reverse_tmdb.tolist()#this is a python list not a json obj
        total = len(json_tmdb)

        response = {
            'total' : total,
            'movieId' : json_tmdb
        }
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug = True )
