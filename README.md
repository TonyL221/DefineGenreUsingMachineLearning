Conclusion of experiment:
Using a neural network to define genre isn't the best because Spotify's algorithm infers data from the audio waveform, conducting inference from an inferred value is not the best. These values were accessed through Spotify's API. There was better accuracy from a linear regression model likely because that's how Spotify defines their genres.
Next time, it will be more consistent if I just do inference from the audio mp3 directly instead of off of Spotify's values, and also maybe scale up the model size.
