import os
import cv2
import imutils
import numpy as np
import threading
from threading import Thread, Event
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from gtts import gTTS
from playsound import playsound

# Use tensorflow.keras for compatibility
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Flatten,
                                     Conv2D, MaxPooling2D, Reshape)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.utils import to_categorical

# Optional: enable GPU memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception as e:
    print("GPU config skipped:", e)

# ------------------------ GUI Setup ------------------------
main = Tk()
main.title("ASL Multi-Model Gesture Recognition")
main.geometry("1300x1200")
# background image optional: keep image file in folder or comment out
try:
    image = Image.open("background.png").resize((1300,1000))
    bg_image = ImageTk.PhotoImage(image)
    background_label = Label(main, image=bg_image)
    background_label.place(relwidth=1, relheight=1)
except Exception:
    # if background not present, ignore
    pass

# ------------------------ Globals ------------------------
names = ['Again','Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes','No Gesture']
classifier = None
playcount = 0
bg = None
X_train = np.array([])
Y_train = np.array([])

# Webcam control
webcam_stop_event = None

# ------------------------ Utility Functions ------------------------
def getID(name):
    try:
        return names.index(name)
    except ValueError:
        return -1

def deleteDirectory():
    os.makedirs('play', exist_ok=True)
    for f in os.listdir('play'):
        if f.endswith('.mp3'):
            try:
                os.remove(os.path.join('play', f))
            except Exception:
                pass

def play(playcount_local, gesture):
    class PlayThread(Thread):
        def __init__(self, playcount, gesture):
            super().__init__()
            self.playcount = playcount
            self.gesture = gesture
        def run(self):
            try:
                os.makedirs('play', exist_ok=True)
                tts = gTTS(text=self.gesture, lang='en', slow=False)
                path = f"play/{self.playcount}.mp3"
                tts.save(path)
                playsound(path)
            except Exception as e:
                print('TTS error:', e)
    PlayThread(playcount_local, gesture).start()

# Background averaging & segmentation (keeps same behavior as original)
def run_avg(image, aWeight=0.5):
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    if bg is None:
        return None
    diff = cv2.absdiff(bg.astype('uint8'), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV compatibility for findContours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) == 0:
        return None
    segmented = max(cnts, key=cv2.contourArea)
    return thresholded, segmented

# ------------------------ Dataset Upload ------------------------
def uploadDataset():
    global X_train, Y_train
    path = filedialog.askdirectory()
    if not path:
        return
    pathlabel.config(text=path)
    Xs = []
    Ys = []
    for root, _, files in os.walk(path):
        label_name = os.path.basename(root)
        label = getID(label_name)
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img = cv2.imread(os.path.join(root, f))
                    img = cv2.resize(img, (28,28))
                    Xs.append(img)
                    Ys.append(label)
                except Exception as e:
                    print('read error', e)
    if len(Xs) == 0:
        text.delete('1.0', END)
        text.insert(END, 'No images found in selected folder.\n')
        return
    X_train = np.array(Xs, dtype='float32') / 255.0
    Y_train = to_categorical(np.array(Ys), num_classes=len(names))
    text.delete('1.0', END)
    text.insert(END, f"Dataset loaded: {len(X_train)} images\n")

# ------------------------ Model Builder ------------------------
def build_model(model_name):
    input_shape = (28,28,3)
    num_classes = len(names)
    if model_name == 'LSTM':
        model = Sequential()
        model.add(Reshape((28*28,3), input_shape=input_shape))
        model.add(LSTM(128))
        model.add(Dense(num_classes, activation='softmax', dtype='float32'))
    elif model_name == 'GRU':
        model = Sequential()
        model.add(Reshape((28*28,3), input_shape=input_shape))
        model.add(GRU(128))
        model.add(Dense(num_classes, activation='softmax', dtype='float32'))
    elif model_name == 'CNN':
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax', dtype='float32'))
    elif model_name == 'CNN_LSTM':
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Reshape((1,32*13*13)))
        model.add(LSTM(64))
        model.add(Dense(num_classes, activation='softmax', dtype='float32'))
    elif model_name in ['GNN','Transformer']:
        text.insert(END, f"{model_name} support not implemented yet\n")
        return None
    else:
        raise ValueError('Unknown model')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------ Training (background thread) ------------------------
def _enable_buttons(enable=True):
    state = 'normal' if enable else 'disabled'
    upload_btn.config(state=state)
    train_btn.config(state=state)
    predict_btn.config(state=state)
    model_selector.config(state=state)

def _training_worker(model_name):
    global classifier
    try:
        classifier = build_model(model_name)
        if classifier is None:
            _enable_buttons(True)
            return
        text.insert(END, f"Starting training: {model_name}\n")
        os.makedirs('model', exist_ok=True)
        checkpoint = ModelCheckpoint(f"model/{model_name}_best.h5", save_best_only=True, monitor='val_loss')
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        csv_logger = CSVLogger(f"model/{model_name}_training_log.csv")
        if X_train.size == 0:
            text.insert(END, 'No dataset loaded. Use Upload Dataset first.\n')
            _enable_buttons(True)
            return
        classifier.fit(X_train, Y_train, batch_size=30, epochs=5, validation_split=0.1,
                       callbacks=[checkpoint, early, reduce_lr, csv_logger], verbose=1)
        classifier.save_weights(f"model/{model_name}_final_weights.h5")
        with open(f"model/{model_name}_model.json", 'w') as f:
            f.write(classifier.to_json())
        text.insert(END, f"Training finished: {model_name}\n")
    except Exception as e:
        text.insert(END, f"Training error: {e}\n")
        print('Training error', e)
    finally:
        _enable_buttons(True)

def start_training():
    _enable_buttons(False)
    model_name = model_selector.get()
    t = Thread(target=_training_worker, args=(model_name,), daemon=True)
    t.start()

# ------------------------ Webcam Prediction (background thread) ------------------------
DEBUG_CONF_THRESHOLD = 0.6

def _webcam_worker(stop_event, conf_threshold=DEBUG_CONF_THRESHOLD, debug=True):
    global playcount, classifier
    if classifier is None:
        text.insert(END, 'No classifier loaded. Train or load a model first.\n')
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        text.insert(END, 'Cannot open camera\n')
        return
    # ROI coords (top, left, bottom, right) - tune for your camera
    top, left, bottom, right = 10, 350, 325, 690
    num_frames = 0
    oldresult = ''
    blur_ksize = (15, 15)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print('Frame not read')
            break
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        roi_color = frame[top:bottom, left:right]
        if roi_color.size == 0:
            if debug:
                print('Empty ROI - check coords')
            cv2.imshow('Video Feed', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set(); break
            continue
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, blur_ksize, 0)
        if num_frames < 30:
            run_avg(roi_gray)
            cv2.putText(display, 'Calibrating background...', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            hand = segment(roi_gray)
            if hand is not None:
                thresholded, segmented = hand
                cv2.drawContours(display, [segmented + (left, top)], -1, (0,0,255), 2)
                img = cv2.resize(roi_color, (28,28)).astype('float32')/255.0
                img = np.expand_dims(img, axis=0)
                try:
                    pred = classifier.predict(img, verbose=0)
                except Exception as e:
                    print('Predict error', e)
                    pred = None
                if pred is not None:
                    prob = float(np.max(pred))
                    pred_idx = int(np.argmax(pred))
                    result = names[pred_idx]
                    if debug:
                        print('pred:', np.round(pred.flatten(), 4))
                        print('max prob:', prob, '->', result)
                    if prob >= conf_threshold:
                        if oldresult != result:
                            oldresult = result
                            try:
                                play(playcount, result)
                                playcount += 1
                            except Exception as e:
                                print('Play error', e)
                        cv2.putText(display, f"{result} ({prob:.2f})", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if debug:
                    cv2.imshow('Thresholded', thresholded)
            else:
                cv2.putText(display, 'No Gesture Detected', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.rectangle(display, (left, top), (right, bottom), (0,255,0), 2)
        cv2.imshow('Video Feed', display)
        num_frames += 1
        # optional: predict every N frames if slow
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set(); break
    cap.release()
    cv2.destroyAllWindows()

def start_webcam_thread():
    global webcam_stop_event
    # if an Event already exists and is a threading.Event and not set -> stop it
    if webcam_stop_event is not None:
        # safe check: use hasattr to avoid AttributeError
        if hasattr(webcam_stop_event, "is_set"):
            if not webcam_stop_event.is_set():
                webcam_stop_event.set()
                webcam_stop_event = None
                return
        else:
            # not a threading.Event (maybe tkinter.Event); replace it
            webcam_stop_event = None

    # create a proper threading.Event
    webcam_stop_event = threading.Event()
    t = Thread(target=_webcam_worker, args=(webcam_stop_event, DEBUG_CONF_THRESHOLD, True), daemon=True)
    t.start()


# ------------------------ Debug Predict Button ------------------------
def debug_predict_samples(n=5):
    global classifier
    if classifier is None:
        text.insert(END, 'No classifier loaded.\n')
        return
    cap = cv2.VideoCapture(0)
    text.insert(END, 'Capturing debug frames...\n')
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            text.insert(END, 'Camera read failed\n'); break
        frame = imutils.resize(frame, width=700); frame = cv2.flip(frame, 1)
        top, left, bottom, right = 10, 350, 325, 690
        roi_color = frame[top:bottom, left:right]
        if roi_color.size == 0:
            text.insert(END, 'Empty ROI - check coords\n'); continue
        img = cv2.resize(roi_color, (28,28)).astype('float32')/255.0
        img = np.expand_dims(img, axis=0)
        pred = classifier.predict(img, verbose=0)
        prob = float(np.max(pred)); idx = int(np.argmax(pred)); label = names[idx]
        text.insert(END, f"Frame {i+1}: {np.round(pred.flatten(),3)} -> {label} ({prob:.3f})\n")
        cv2.imshow(f"Debug Frame {i+1}", roi_color)
        cv2.waitKey(300)
    cap.release(); cv2.destroyAllWindows()

# ------------------------ GUI Buttons ------------------------
font1 = ('times', 13, 'bold')
upload_btn = Button(main, text='Upload Dataset', command=uploadDataset)
upload_btn.place(x=50, y=300); upload_btn.config(font=font1)
train_btn = Button(main, text='Train Selected Model', command=start_training)
train_btn.place(x=50, y=350); train_btn.config(font=font1)
predict_btn = Button(main, text='Predict Webcam', command=start_webcam_thread)
predict_btn.place(x=50, y=400); predict_btn.config(font=font1)

model_selector = ttk.Combobox(main, values=['LSTM','GRU','CNN','CNN_LSTM','GNN','Transformer'])
model_selector.place(x=50, y=250)
model_selector.current(0)

pathlabel = Label(main, text='', bg='yellow', fg='black')
pathlabel.place(x=50, y=220)

text = Text(main, height=12, width=78)
text.place(x=400, y=300)

# debug predict button
debug_btn = Button(main, text='Debug Predict (5)', command=debug_predict_samples)
debug_btn.place(x=50, y=450); debug_btn.config(font=font1)

# cleanup and start
deleteDirectory()
main.mainloop()
