# cnn-lstm
CNN-LSTM for creating a latent representation of KTH action dataset videos

Decide your own:  No. of frames from each video
                  No. of videos from each class
                  Parameters for the model
                  
Canny edge detection helps remove empty frames

All frames were resized to 128x128 dimension due to errors in model

npy_dataset.py creates separate numpy files for each action class and returns file of dimension (no.of vids, no.of frames, 128, 128, 1)

cnn_lstm.py takes each action class numpy file separately and makes predictions

I have not uploaded the numpy files, create your own

Feel free to download and modify
