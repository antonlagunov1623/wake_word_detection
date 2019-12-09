from keras.models import Model
from keras.layers import Input, Permute, Reshape, Lambda, Dot, Softmax
from keras.layers import Add, BatchNormalization, Conv2D, Dense, LSTM, Bidirectional
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

def AttRNNSpeechModel(samplingrate = 16000, inputLength = 16000, rnn_func = LSTM):
    sr = samplingrate
    iLen = inputLength
    
    inputs = Input((inputLength,), name='input')

    x = Reshape((1, -1)) (inputs)

    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                             padding='same', sr=sr, n_mels=80,
                             fmin=40.0, fmax=sr/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_stft') (x)

    x = Normalization2D(int_axis=0)(x)

    #note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    #we would rather have it the other way around for LSTMs

    x = Permute((2,1,3)) (x)

    x = Conv2D(10, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(1, (5,1) , activation='relu', padding='same') (x)
    x = BatchNormalization() (x)

    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim') (x) #keras.backend.squeeze(x, axis)

    x = Bidirectional(rnn_func(64, return_sequences = True, dropout=0.3)) (x) # [b_s, seq_len, vec_dim]
    x = Bidirectional(rnn_func(64, return_sequences = True, dropout=0.3)) (x) # [b_s, seq_len, vec_dim]

    xFirst = Lambda(lambda q: q[:,64]) (x) #[b_s, vec_dim]
    query = Dense(128) (xFirst)

    #dot product attention
    attScores = Dot(axes=[1,2])([query, x]) 
    attScores = Softmax(name='attSoftmax')(attScores) #[b_s, seq_len]

    #rescale sequence
    attVector = Dot(axes=[1,1])([attScores, x]) #[b_s, vec_dim]

    x = Dense(64, activation = 'relu')(attVector)
    x = Dense(32)(x)

    output = Dense(1, activation = 'sigmoid', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model