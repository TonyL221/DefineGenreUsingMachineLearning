import numpy as np
from keras.models import load_model
from functs import UsefulFunctions
uf = UsefulFunctions()

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

array = uf.get_audio_features(uf.cleanUpLink('https://open.spotify.com/track/2AZXe8nq18mPITmPy5KJwQ?si=3f67659ca9d14a9e'))#replace link with track
print(array)
#prediction for random track
new_data = np.array([array])#replace with array 
'''
# Fit and transform your data
array_scaled = scaler.fit_transform(new_data)

#
print("\nScaled Input Data:")
print(array_scaled)'''
array_scaled = new_data
# Load the saved model
loaded_model = load_model('modelnew.keras')

predictions = loaded_model.predict(array_scaled)

predicted_class_index = np.argmax(predictions)
genre_mapping = {0: 'Pop', 1: 'Rap', 2: 'EDM', 3: 'Rock', 4: 'Chill'}
predicted_genre = genre_mapping[predicted_class_index]

print("Predicted Genre:", predicted_genre)
print("Predictions: ", predictions)

