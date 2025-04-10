# from datetime import datetime , timedelta 
# from flask import render_template, redirect , request , url_for , flash
# # from app import app , db , os 
# from app import app  , os 

# from textblob import TextBlob
# # from app.models import User, Archivist , Section , Feedback , Book , user_book_association

# import os
# import numpy as np
# from random import choice
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# os.environ['TF_ENABLE_ONEDNN_QPTS'] = '0'

# amount_earned = 0 
# USER_ID = 0
# ARCHIVIST_ID = 0


# emotion_to_spotify_link = {
#     "Angry": [
#         "https://open.spotify.com/episode/5CBvXPfKNHVbLF4EG1XufX?si=143428f7f3334e0c",  
#         "https://open.spotify.com/episode/7aKCmGHBDFmffbL1NjNUfo?si=9f88257b50d041f1"
#     ],
#     "Disgusted": [
#         "https://open.spotify.com/episode/50Ci7SHw7UlQGQhQd8zVgE?si=0bfed9d2120d4a06",  
#         "https://open.spotify.com/episode/6gVvoMxepbTvjA4ZNJLNrz?si=f0b4d34473394a29"
#     ],
#     "Fearful": [
#         "https://open.spotify.com/episode/0TdPnBwvb2DHu7wCq3RhHo?si=f9b7a61aac2c4525",  
#         "https://open.spotify.com/episode/7nkQ93hrJGLiRQt4bJgvq1?si=ef02198288a34d3c"
#     ],
#     "Happy": [
#         "https://open.spotify.com/track/0fJH2SsaZ2N619cdkkhN1o?si=e53bb561e45649c6",  
#         "https://open.spotify.com/episode/3pgS23c6tPOuBSqHOMWjQO?si=92547b9ee0b74f69",
#         "https://open.spotify.com/episode/6VMgvrmzQRetQvcfwVYaCC?si=c582945e0db44007"
#     ],
#     "Neutral": [
#         "https://open.spotify.com/episode/2xT7glH4046UZoWJPK0DAf?si=06967041b5944433",  
#         "https://open.spotify.com/episode/5dqj3jPNqToxvlsKWbUa9V?si=bc8662d6a19b4818"
#     ],
#     "Sad": [
#         "https://open.spotify.com/episode/4Rnv8yGQ5cnl4yXO7hvFW4?si=b39b55b01182422e",  
#         "https://open.spotify.com/episode/6I3ITgkVYUZRBrKK2GGrCO?si=2e817968ef2e45ce",
#         "https://open.spotify.com/playlist/4kja2U12BZYMicZIDljzWT?si=Vvlig9QjRk65JXMuoPUiCQ"
#     ],
#     "Surprised": [
#         "https://open.spotify.com/episode/0Sk836twiPIIMg8wugePCO?si=317ef3be96004b58",  
#         "https://open.spotify.com/episode/4LIFfyKjeBqgUvMxYdGKmp?si=6a98fa7751f34170",
#         "https://open.spotify.com/playlist/37i9dQZF1DX54H77RAFJQ9?si=T2uMvU_zSG-ewocRuINNfg"
#     ]
# }


# #HOME PAGE
# @app.route("/")
# def hello():
# 	return render_template("user_dashboard.html" , title = "Home" )




# @app.route( "/blob"  )
# def blob() :

#     model = Sequential()

#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))

#     current_dir = os.path.dirname(os.path.realpath(__file__))
#     model_path = os.path.join(current_dir, 'model.h5')
    
#     model.load_weights( model_path )

#     # prevents openCL usage and unnecessary logging messages
#     cv2.ocl.setUseOpenCL(False)

#     # dictionary which assigns each label an emotion (alphabetical order)
#     emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#     # start the webcam feed
#     cap = cv2.VideoCapture(0)
#     maxindex = 0 
#     while True:
#         # Find haar cascade to draw bounding box around face
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         current_dir = os.path.dirname(os.path.realpath(__file__))
#         model_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
        
#         facecasc = cv2.CascadeClassifier( model_path )
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#             prediction = model.predict(cropped_img)

#             maxindex = int(np.argmax(prediction))
#             cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow('Video', cv2.resize(frame,(1600,960), interpolation = cv2.INTER_CUBIC))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



#     given_category = emotion_dict[maxindex]

#     # Choose a random audiobook link from the mapped emotion
#     random_audio_link = choice(emotion_to_spotify_link[given_category])

#     # Redirect the user to the selected Spotify audiobook link
#     return redirect(random_audio_link)
#     # given_category = emotion_dict[maxindex]
#     # recommended_books = []

#     # books = Book.query.all()
#     # for book in books:
#     #     blob = TextBlob( book.content )

#     #     polarity = blob.sentiment.polarity

#     #     if polarity <= -0.6:
#     #         category = 'Angry'
#     #     elif -0.6 < polarity <= -0.34:
#     #         category = 'Disgust'
#     #     elif -0.34 < polarity <= -0.15:
#     #          category = 'Fear'
#     #     elif -0.15 < polarity < 0 :
#     #         category = 'Sad'
#     #     elif 0.20 <= polarity <= 0.50 :
#     #         category = 'Surprised'
#     #     elif  polarity > 0.50 :
#     #         category = 'Happy'
#     #     else:
#     #         category = 'Neutral'

#     #     if category == given_category :
#     #          recommended_books.append( book )

#     # section_ids = []
#     # for book in recommended_books:
#     #     section_ids.append(book.section.id)

#     # section_names = []
#     # for id in section_ids:
#     #     section = Section.query.filter_by( id = id ).first()
#     #     section_names.append(section.name)         
          
#     # return render_template( 'recommendation.html' , recommendation = recommended_books , sections = section_names  )      


# @app.route( "/four_not_four" )
# def four_not_four():
# 	return render_template( 'four_not_four.html' )


# //////////////presentation day

# from flask import render_template, redirect, request, jsonify
# from app import app
# import os
# import numpy as np
# import cv2
# from random import choice
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# import tempfile

# # Disable TensorFlow warnings
# os.environ['TF_ENABLE_ONEDNN_QPTS'] = '0'

# # Emotion-based recommendations
# emotion_recommendations = {
#     "Angry": {
#         "songs": [
#             "https://open.spotify.com/track/4UtAMg91FQ5AwF4lZMStQy?si=fa238c52e0e84b1d",  
#             "https://open.spotify.com/track/5AsHQzHhLJFIvE8l2OdOoG?si=ad3863e6fe3242d5"
#         ],
#         "books": [
#             "https://open.spotify.com/episode/5CBvXPfKNHVbLF4EG1XufX?si=143428f7f3334e0c",  
#             "https://open.spotify.com/episode/7aKCmGHBDFmffbL1NjNUfo?si=9f88257b50d041f1"
#         ]
#     },
#     "Happy": {
#         "songs": [
#             "https://open.spotify.com/track/4gxTJiCBfQUpqtWeSQCZcK?si=b335a6e500aa4dac",  
#             "https://open.spotify.com/track/5fwRhPXAvGgzUxW62ZCvEg?si=03b915292ef74146"
#         ],
#         "books": [
#             "https://open.spotify.com/track/0fJH2SsaZ2N619cdkkhN1o?si=e53bb561e45649c6",  
#             "https://open.spotify.com/episode/3pgS23c6tPOuBSqHOMWjQO?si=92547b9ee0b74f69",
#             "https://open.spotify.com/episode/6VMgvrmzQRetQvcfwVYaCC?si=c582945e0db44007"
#         ]
#     },
#     "Neutral": {
#         "songs": [
#             "https://open.spotify.com/track/4NmWv6PEGEAmorN5UN9fRo?si=b0ccfe9ce3f1469c",  
#             "https://open.spotify.com/track/2k0m3S5y6J3RPBEwrYAHrU?si=d962b2b9438f41cf"
#         ],
#         "books": [
#             "https://open.spotify.com/episode/2xT7glH4046UZoWJPK0DAf?si=06967041b5944433",  
#             "https://open.spotify.com/episode/3pgS23c6tPOuBSqHOMWjQO?si=92547b9ee0b74f69",
#             "https://open.spotify.com/episode/5dqj3jPNqToxvlsKWbUa9V?si=bc8662d6a19b4818"
#         ]
#     },
#     "Sad": {
#         "songs": [
#             "https://open.spotify.com/track/4NmWv6PEGEAmorN5UN9fRo?si=b0ccfe9ce3f1469c",  
#             "https://open.spotify.com/track/2k0m3S5y6J3RPBEwrYAHrU?si=d962b2b9438f41cf",
#             "https://open.spotify.com/track/6tulU5hdoXmkeM0e8JOGz3?si=dfcaf51a611f4755"
#         ],
#         "books": [
#             "https://open.spotify.com/episode/4Rnv8yGQ5cnl4yXO7hvFW4?si=b39b55b01182422e",  
#             "https://open.spotify.com/episode/6I3ITgkVYUZRBrKK2GGrCO?si=2e817968ef2e45ce",
#         ]
#     },
#     "Fearful": {
#         "songs": [
#             "https://open.spotify.com/track/66aRgKCtqQFWLSGeFKgWQt?si=e5f28bae64724efd",  
#             "https://open.spotify.com/track/2rWYfMbQgHYXPEzbhN65Ys?si=5a1ab818f8c24bb6",
#         ],
#         "books": [
#             "https://open.spotify.com/episode/0TdPnBwvb2DHu7wCq3RhHo?si=f9b7a61aac2c4525",  
#             "https://open.spotify.com/episode/7nkQ93hrJGLiRQt4bJgvq1?si=ef02198288a34d3c",
#         ]
#     },
#      "Surprised": {
#         "songs": [
#             "https://open.spotify.com/track/4blqlsA1uf2d2I40E90EUC?si=49f2daee276c4e84",  
#             "https://open.spotify.com/track/04ChFQZCKl2yRcvc0VR7qN?si=59b1f29bf95c4403",
#         ],
#         "books": [
#             "https://open.spotify.com/episode/0Sk836twiPIIMg8wugePCO?si=317ef3be96004b58",  
#             "https://open.spotify.com/episode/4LIFfyKjeBqgUvMxYdGKmp?si=6a98fa7751f34170",
#         ]
#     },
#      "Disgusted": {
#         "songs": [
#             "https://open.spotify.com/track/73virUhdH3B8pi53zaG2BL?si=bdea2584d8a14d28",  
#             "https://open.spotify.com/track/2hs0eVQRkBiGw8nLPN2rIu?si=26cb51eabfd9459a",
#         ],
#         "books": [
#             "https://open.spotify.com/episode/50Ci7SHw7UlQGQhQd8zVgE?si=0bfed9d2120d4a06",  
#             "https://open.spotify.com/episode/6gVvoMxepbTvjA4ZNJLNrz?si=f0b4d34473394a29",
#         ]
#     },
    
#     # Add other emotions with respective songs and books
# }

# @app.route("/")
# def home():
#     return render_template("webcam.html", title="Emotion Detection")

# @app.route("/analyze_emotion", methods=["POST"])
# def analyze_emotion():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400
    
#     file = request.files['image']
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#     file.save(temp_file.name)
    
#     model = load_emotion_model()
#     emotion = detect_emotion(model, temp_file.name)
#     os.unlink(temp_file.name)
    
#     recommendations = emotion_recommendations.get(emotion, {"songs": [], "books": []})
#     recommendation_type = choice(["songs", "books"]) if recommendations["songs"] and recommendations["books"] else "songs" if recommendations["songs"] else "books"
#     recommendation = choice(recommendations[recommendation_type]) if recommendations[recommendation_type] else "No recommendation available"
    
#     return jsonify({
#         "emotion": emotion,
#         "recommendation_type": recommendation_type,
#         "recommendation": recommendation
#     })

# def load_emotion_model():
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
    
#     model.load_weights(os.path.join(os.path.dirname(__file__), 'model.h5'))
#     return model

# def detect_emotion(model, image_path):
#     emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#     frame = cv2.imread(image_path)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
#     facecasc = cv2.CascadeClassifier(cascade_path)
#     faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
#     if len(faces) == 0:
#         return "Neutral"
    
#     x, y, w, h = faces[0]
#     roi_gray = gray[y:y + h, x:x + w]
#     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#     prediction = model.predict(cropped_img)
#     maxindex = int(np.argmax(prediction))
    
#     return emotion_dict[maxindex]

# @app.route("/four_not_four")
# def four_not_four():
#     return render_template('four_not_four.html')


# ///////////////


from flask import render_template, redirect, request, jsonify
from app import app
import os
import numpy as np
import cv2
from random import choice
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tempfile

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_QPTS'] = '0'

# Emotion-based recommendations
emotion_recommendations = {
    "Angry": {
        "songs": [
            "https://open.spotify.com/track/4UtAMg91FQ5AwF4lZMStQy?si=fa238c52e0e84b1d",  
            "https://open.spotify.com/track/5AsHQzHhLJFIvE8l2OdOoG?si=ad3863e6fe3242d5"
        ],
        "books": [
            "https://open.spotify.com/episode/5CBvXPfKNHVbLF4EG1XufX?si=143428f7f3334e0c",  
            "https://open.spotify.com/episode/7aKCmGHBDFmffbL1NjNUfo?si=9f88257b50d041f1"
        ]
    },
    "Happy": {
        "songs": [
            "https://open.spotify.com/track/4gxTJiCBfQUpqtWeSQCZcK?si=b335a6e500aa4dac",  
            "https://open.spotify.com/track/5fwRhPXAvGgzUxW62ZCvEg?si=03b915292ef74146"
        ],
        "books": [
            "https://open.spotify.com/track/0fJH2SsaZ2N619cdkkhN1o?si=e53bb561e45649c6",  
            "https://open.spotify.com/episode/3pgS23c6tPOuBSqHOMWjQO?si=92547b9ee0b74f69",
            "https://open.spotify.com/episode/6VMgvrmzQRetQvcfwVYaCC?si=c582945e0db44007"
        ]
    },
    "Neutral": {
        "songs": [
            "https://open.spotify.com/track/4NmWv6PEGEAmorN5UN9fRo?si=b0ccfe9ce3f1469c",  
            "https://open.spotify.com/track/2k0m3S5y6J3RPBEwrYAHrU?si=d962b2b9438f41cf"
        ],
        "books": [
            "https://open.spotify.com/episode/2xT7glH4046UZoWJPK0DAf?si=06967041b5944433",  
            "https://open.spotify.com/episode/3pgS23c6tPOuBSqHOMWjQO?si=92547b9ee0b74f69",
            "https://open.spotify.com/episode/5dqj3jPNqToxvlsKWbUa9V?si=bc8662d6a19b4818"
        ]
    },
    "Sad": {
        "songs": [
            "https://open.spotify.com/track/4NmWv6PEGEAmorN5UN9fRo?si=b0ccfe9ce3f1469c",  
            "https://open.spotify.com/track/2k0m3S5y6J3RPBEwrYAHrU?si=d962b2b9438f41cf",
            "https://open.spotify.com/track/6tulU5hdoXmkeM0e8JOGz3?si=dfcaf51a611f4755"
        ],
        "books": [
            "https://open.spotify.com/episode/4Rnv8yGQ5cnl4yXO7hvFW4?si=b39b55b01182422e",  
            "https://open.spotify.com/episode/6I3ITgkVYUZRBrKK2GGrCO?si=2e817968ef2e45ce",
        ]
    },
    "Fearful": {
        "songs": [
            "https://open.spotify.com/track/66aRgKCtqQFWLSGeFKgWQt?si=e5f28bae64724efd",  
            "https://open.spotify.com/track/2rWYfMbQgHYXPEzbhN65Ys?si=5a1ab818f8c24bb6",
        ],
        "books": [
            "https://open.spotify.com/episode/0TdPnBwvb2DHu7wCq3RhHo?si=f9b7a61aac2c4525",  
            "https://open.spotify.com/episode/7nkQ93hrJGLiRQt4bJgvq1?si=ef02198288a34d3c",
        ]
    },
     "Surprised": {
        "songs": [
            "https://open.spotify.com/track/4blqlsA1uf2d2I40E90EUC?si=49f2daee276c4e84",  
            "https://open.spotify.com/track/04ChFQZCKl2yRcvc0VR7qN?si=59b1f29bf95c4403",
        ],
        "books": [
            "https://open.spotify.com/episode/0Sk836twiPIIMg8wugePCO?si=317ef3be96004b58",  
            "https://open.spotify.com/episode/4LIFfyKjeBqgUvMxYdGKmp?si=6a98fa7751f34170",
        ]
    },
     "Disgusted": {
        "songs": [
            "https://open.spotify.com/track/73virUhdH3B8pi53zaG2BL?si=bdea2584d8a14d28",  
            "https://open.spotify.com/track/2hs0eVQRkBiGw8nLPN2rIu?si=26cb51eabfd9459a",
        ],
        "books": [
            "https://open.spotify.com/episode/50Ci7SHw7UlQGQhQd8zVgE?si=0bfed9d2120d4a06",  
            "https://open.spotify.com/episode/6gVvoMxepbTvjA4ZNJLNrz?si=f0b4d34473394a29",
        ]
    },
    
    # Add other emotions with respective songs and books
}

@app.route("/")
def home():
    return render_template("webcam.html", title="Emotion Detection")

@app.route("/analyze_emotion", methods=["POST"])
def analyze_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    content_type = request.form.get('content_type', 'song')  # Get the content type from form data
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)
    
    model = load_emotion_model()
    emotion = detect_emotion(model, temp_file.name)
    os.unlink(temp_file.name)
    
    recommendations = emotion_recommendations.get(emotion, {"songs": [], "books": []})
    
    # Map the frontend content_type to the backend category
    recommendation_type = "songs" if content_type == "song" else "books"
    
    # Check if recommendations exist for this type
    if not recommendations[recommendation_type]:
        # Fall back to the other type if no recommendations exist for the requested type
        recommendation_type = "books" if recommendation_type == "songs" else "songs"
    
    recommendation = choice(recommendations[recommendation_type]) if recommendations[recommendation_type] else "No recommendation available"
    
    return jsonify({
        "emotion": emotion,
        "recommendation_type": "song" if recommendation_type == "songs" else "audiobook",
        "recommendation": recommendation
    })
    
def load_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    model.load_weights(os.path.join(os.path.dirname(__file__), 'model.h5'))
    return model

def detect_emotion(model, image_path):
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    facecasc = cv2.CascadeClassifier(cascade_path)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return "Neutral"
    
    x, y, w, h = faces[0]
    roi_gray = gray[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    
    return emotion_dict[maxindex]

@app.route("/four_not_four")
def four_not_four():
    return render_template('four_not_four.html')



