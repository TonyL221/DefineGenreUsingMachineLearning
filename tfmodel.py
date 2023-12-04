import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense, Dropout,BatchNormalization, Flatten,LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Load your dataset
df = pd.read_csv('Track_audio_features and genre - Sheet1.csv')

# Features and labels
X = df.drop(columns=['Genre'])
y = df['Genre']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X.values[0][0])#figure out how to convert from [ v v v] to [1,2,3]

####debugging
#print("\nscaled data: ",X_train,y_train)
# Define the model
model = Sequential()
#Add dense layer, ReLU, 1st layer
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#Batch Normalize 
model.add(BatchNormalization())
#Dropout to prevent overfitting
model.add(Dropout(0.55))

#Add dense layer, ReLU, 2nd layer
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#Batch Normalize 
model.add(BatchNormalization())
#Dropout to prevent overfitting
model.add(Dropout(0.55))

#Add dense layer, ReLU, 3rd layer
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#Dropout to prevent overfitting
model.add(Dropout(0.55))

#Add dense layer, ReLU, 4th layer
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
#Dropout to prevent overfitting
model.add(Dropout(0.55))
model.add(BatchNormalization())


#Output Layer
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
from keras.optimizers import Adam
optimizer = Adam(learning_rate=.00002)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_accuracy', patience=30)#maybe use this later

history =model.fit(X_train, y_train, epochs=256, batch_size=80, validation_split=0.2)

'''# Plot the loss over time
plt.plot(history.history['loss'])
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()'''


# Plot the accuracy over time
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')


model.save('modelnew.keras')