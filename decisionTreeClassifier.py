#import stuff
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tools
#get the data
df = pd.read_csv('Track_audio_features and genre - Sheet1.csv')#replace with better data

#print(df.shape)
#create the prediction column and the data column
X = df.drop(columns=['Genre'])
Y = df['Genre']
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=.8)

#create object 
model = DecisionTreeClassifier()

#train the model
model.fit(X_train,Y_train)

#accuracy score
print("Accuracy: ",accuracy_score(Y_test, model.predict(X_test))*100,"%")

arr = tools.get_audio_features(tools.cleanUpLink('https://open.spotify.com/track/7gjzVKoft49SSuQc7BLLQI?si=7ec319b79c734794'))
new_song_features_reshaped = [arr]
# Use the trained model to predict the genre
predicted_genre = model.predict(new_song_features_reshaped)

print("Predicted Genre:", predicted_genre)