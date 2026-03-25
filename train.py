# train.py - updated to include RESNET_SMALL and MOBILENET_LIKE
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Activation, Flatten,
    LSTM, GRU, Conv2D, MaxPooling2D, BatchNormalization,
    Input, TimeDistributed, GlobalAveragePooling1D,
    Add, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling2D
)
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

# ------------------- CONFIG -------------------
DATA_PATH = "Dataset"
LABELS = ['Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes', 'ZNo Gesture']
IMG_H, IMG_W = 28, 28
BATCH_SIZE = 32
EPOCHS = 25
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- HELPERS -------------------
def getID(name):
    return LABELS.index(name) if name in LABELS else 0

# ------------------- LOAD DATA -------------------
X, Y = [], []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'thumbs.db' not in file.lower():
            img = cv2.imread(os.path.join(root, file))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_W, IMG_H))
            X.append(img)
            Y.append(getID(os.path.basename(root)))
            print(f"Loaded: {os.path.basename(root)}/{file} -> {getID(os.path.basename(root))}")

if len(X) == 0:
    raise SystemExit("No images found in Dataset directory. Check 'Dataset' and subfolders named exactly as LABELS.")

X = np.asarray(X, dtype='float32') / 255.0
Y = to_categorical(np.asarray(Y))
print("X shape:", X.shape, "Y shape:", Y.shape)

# shuffle and split
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
num_classes = Y.shape[1]

# sequence versions (for LSTM/GRU/TRANSFORMER)
seq_timesteps = IMG_H
seq_features = IMG_W * 3
X_seq_train = X_train.reshape((X_train.shape[0], seq_timesteps, seq_features))
X_seq_test  = X_test.reshape((X_test.shape[0], seq_timesteps, seq_features))

# ------------------- MODEL BUILDERS -------------------
def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_rnn_flat(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_lstm(seq_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=seq_shape))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_gru(seq_shape, num_classes):
    model = Sequential()
    model.add(GRU(128, input_shape=seq_shape))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_transformer_like(seq_shape, num_classes):
    inputs = Input(shape=seq_shape)
    x = TimeDistributed(Dense(128, activation='relu'))(inputs)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- NEW: small residual block
def res_block(x, filters, kernel=(3,3)):
    y = Conv2D(filters, kernel, padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = Conv2D(filters, kernel, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def build_resnet_small(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # pad channels to match residual path
    x = Conv2D(32, (1,1), padding='same')(x)
    x = res_block(x, 32)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = res_block(x, 64)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

# --- NEW: MobileNet-like using SeparableConv2D (lightweight)
def build_mobilenet_like(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)

    x = SeparableConv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = SeparableConv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

# ------------------- TRAIN LOOP -------------------
models_to_train = [
    ("RNN", lambda: build_rnn_flat((IMG_H, IMG_W, 3), num_classes), "image"),
    ("CNN", lambda: build_cnn((IMG_H, IMG_W, 3), num_classes), "image"),
    ("RESNET_SMALL", lambda: build_resnet_small((IMG_H, IMG_W, 3), num_classes), "image"),
    ("MOBILENET_LIKE", lambda: build_mobilenet_like((IMG_H, IMG_W, 3), num_classes), "image"),
    ("LSTM", lambda: build_lstm((seq_timesteps, seq_features), num_classes), "seq"),
    ("GRU", lambda: build_gru((seq_timesteps, seq_features), num_classes), "seq"),
    ("TRANSFORMER", lambda: build_transformer_like((seq_timesteps, seq_features), num_classes), "seq")
]

for name, builder, data_mode in models_to_train:
    print("\n" + "-"*44)
    print(f"Training {name} (mode={data_mode})")
    model = builder()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    if data_mode == "image":
        x_tr, x_val = X_train, X_test
    else:
        x_tr, x_val = X_seq_train, X_seq_test

    hist = model.fit(x_tr, y_train, validation_data=(x_val, y_test),
                     batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

    weights_path = os.path.join(MODEL_DIR, f"{name}_weights.h5")
    json_path = os.path.join(MODEL_DIR, f"{name}_model.json")
    history_path = os.path.join(MODEL_DIR, f"{name}_history.pckl")

    model.save_weights(weights_path)
    with open(json_path, "w") as jf:
        jf.write(model.to_json())
    with open(history_path, "wb") as hf:
        pickle.dump(hist.history, hf)

    print(f"{name} saved -> {weights_path}, {json_path}, {history_path}")

print("\nAll done.")
