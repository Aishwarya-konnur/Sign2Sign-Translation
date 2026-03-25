# HandGestureRecognize.py - updated to support RESNET_SMALL & MOBILENET_LIKE
import os
import cv2
import numpy as np
import tkinter
from tkinter import *
from tkinter import filedialog
from keras.models import model_from_json
import pickle
import imutils
from gtts import gTTS
from playsound import playsound
from threading import Thread
from PIL import Image, ImageTk

MODEL_DIR = "model"
IMG_H, IMG_W = 28, 28
NAMES = ['Bye', 'Hello', 'No', 'Perfect', 'Thank You', 'Yes', 'ZNo Gesture']
MODEL_KEYS = ["RNN", "CNN", "RESNET_SMALL", "MOBILENET_LIKE", "LSTM", "GRU", "TRANSFORMER"]
PRED_THRESHOLD = 0.99
os.makedirs("play", exist_ok=True)

classifier = None
current_model_key = None
playcount = 0
bg_accum = None

main = tkinter.Tk()
main.title("ASL Recognition")
main.geometry("1300x1200")

try:
    image = Image.open("background.png")
    image = image.resize((1300, 1000), Image.ANTIALIAS)
    bg_image = ImageTk.PhotoImage(image)
    background_label = tkinter.Label(main, image=bg_image)
    background_label.place(relwidth=1, relheight=1)
except Exception:
    main.config(bg='magenta3')

font1 = ('times', 13, 'bold')
Label(main, text="Select Model:", bg='yellow4', fg='black', font=font1).place(x=50, y=200)
model_var = StringVar(main)
model_var.set(MODEL_KEYS[0])
OptionMenu(main, model_var, *MODEL_KEYS).place(x=50, y=230)

load_button = Button(main, text="Load Selected Model", font=font1)
load_button.place(x=50, y=260)

upload_btn = Button(main, text="Upload ASL Dataset (optional)", font=font1)
upload_btn.place(x=50, y=300)

pathlabel = Label(main, bg='yellow4', fg='white', font=font1)
pathlabel.place(x=50, y=350)

train_button = Button(main, text="(Optional) Run train.py", font=font1)
train_button.place(x=50, y=400)

predict_button = Button(main, text="ASL Recognition from Webcam", font=font1)
predict_button.place(x=50, y=450)

text = Text(main, height=12, width=78, font=('times', 12, 'bold'))
text.place(x=450, y=300)

def show_msg(s):
    text.insert(END, s + "\n")
    text.see(END)

def delete_play_mp3s():
    for f in os.listdir("play"):
        if f.endswith(".mp3"):
            try:
                os.remove(os.path.join("play", f))
            except:
                pass

def tts_play(cnt, gesture):
    class TTSPlay(Thread):
        def __init__(self, cnt, text_to_say):
            Thread.__init__(self)
            self.cnt = cnt
            self.text_to_say = text_to_say
        def run(self):
            try:
                fname = os.path.join("play", f"{self.cnt}.mp3")
                t1 = gTTS(text=self.text_to_say, lang='en', slow=False)
                t1.save(fname)
                playsound(fname)
            except Exception as e:
                print("TTS error:", e)
    thr = TTSPlay(cnt, gesture)
    thr.start()

def load_model_for_key(key):
    json_path = os.path.join(MODEL_DIR, f"{key}_model.json")
    weights_path = os.path.join(MODEL_DIR, f"{key}_weights.h5")
    hist_path = os.path.join(MODEL_DIR, f"{key}_history.pckl")
    if not os.path.exists(json_path): raise FileNotFoundError(f"Missing {json_path}")
    if not os.path.exists(weights_path): raise FileNotFoundError(f"Missing {weights_path}")
    with open(json_path, 'r') as jf:
        model_json = jf.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = None
    if os.path.exists(hist_path):
        try:
            with open(hist_path, 'rb') as hf:
                hist = pickle.load(hf)
        except:
            hist = None
    return model, hist

def upload_dataset():
    folder = filedialog.askdirectory(initialdir=".")
    if folder:
        pathlabel.config(text=folder)
        show_msg(f"Dataset folder selected: {folder}")
        try:
            groups = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            show_msg("Classes found: " + ", ".join(groups))
        except Exception as e:
            show_msg("Error inspecting dataset: " + str(e))

def run_train_py():
    import subprocess, threading, sys
    def run_and_stream():
        cmd = [sys.executable, "train.py"]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        except Exception as e:
            show_msg("Failed to start train.py: " + str(e))
            return
        show_msg("train.py started...")
        for line in p.stdout:
            show_msg(line.rstrip())
        p.wait()
        show_msg(f"train.py exited with code {p.returncode}")
    threading.Thread(target=run_and_stream, daemon=True).start()

def load_selected_model():
    global classifier, current_model_key
    text.delete('1.0', END)
    key = model_var.get()
    show_msg(f"Loading model '{key}' ...")
    try:
        classifier, hist = load_model_for_key(key)
        current_model_key = key
        show_msg(f"Loaded model '{key}'.")
        if hist and 'accuracy' in hist:
            try:
                show_msg(f"Last training accuracy: {hist['accuracy'][-1]:.4f}")
            except: pass
        if key in ("CNN", "RNN", "RESNET_SMALL", "MOBILENET_LIKE"):
            show_msg("Model expects IMAGE input of shape (1,28,28,3).")
        else:
            show_msg("Model expects SEQUENCE input (1, timesteps=28, features=28*3).")
    except Exception as e:
        classifier = None
        current_model_key = None
        show_msg("Error loading model: " + str(e))

def webcam_predict():
    global playcount, classifier, current_model_key, bg_accum
    if classifier is None:
        show_msg("Load a model first.")
        return
    old_result = None
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        show_msg("Cannot open camera.")
        return
    fgbg = cv2.createBackgroundSubtractorKNN()
    top, right, bottom, left = 10, 350, 325, 690
    num_frames = 0
    bg_accum = None
    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            show_msg("Camera read failed.")
            break
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (41, 41), 0)
        if num_frames < 30:
            if bg_accum is None:
                bg_accum = gray.copy().astype("float")
            else:
                cv2.accumulateWeighted(gray, bg_accum, 0.5)
        else:
            diff = cv2.absdiff(bg_accum.astype("uint8"), gray)
            thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) != 0:
                segmented = max(cnts, key=cv2.contourArea)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                roi_color = frame[top:bottom, right:left]
                roi_masked = fgbg.apply(roi_color)
                cv2.imwrite("temp_test.jpg", roi_masked)
                img = cv2.imread("temp_test.jpg")
                if img is None:
                    show_msg("Error reading temp image.")
                    break
                img = cv2.resize(img, (IMG_W, IMG_H))
                img = img.astype('float32') / 255.0
                try:
                    if current_model_key in ("CNN", "RNN", "RESNET_SMALL", "MOBILENET_LIKE"):
                        inp = img.reshape(1, IMG_H, IMG_W, 3)
                    else:
                        seq = img.reshape(IMG_H, IMG_W * 3)
                        inp = seq.reshape(1, IMG_H, IMG_W * 3)
                except Exception as e:
                    show_msg("Preprocess error: " + str(e))
                    inp = None
                if inp is not None:
                    preds = classifier.predict(inp)
                    val = float(np.max(preds))
                    idx = int(np.argmax(preds))
                    pred_label = NAMES[idx] if idx < len(NAMES) else str(idx)
                    if val >= PRED_THRESHOLD:
                        cv2.putText(clone, f"Gesture: {pred_label} ({val:.2f})", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        if old_result != pred_label:
                            tts_play(playcount, pred_label)
                            isl_path = os.path.join("ISL", pred_label + ".jpg")
                            if os.path.exists(isl_path):
                                isl_img = cv2.imread(isl_path)
                                isl_img = cv2.resize(isl_img, (300,300))
                                cv2.imshow("ISL Output", isl_img)
                            asl_path = os.path.join("ASL", pred_label + ".jpg")
                            if os.path.exists(asl_path):
                                asl_img = cv2.imread(asl_path)
                                asl_img = cv2.resize(asl_img, (300,300))
                                cv2.imshow("ASL Output", asl_img)
                            old_result = pred_label
                            playcount += 1
                    else:
                        cv2.putText(clone, "No confident gesture", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.imshow("Gesture Region (binary)", roi_masked)
            else:
                cv2.putText(clone, "No Gesture is Shown", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

load_button.config(command=load_selected_model)
upload_btn.config(command=upload_dataset)
train_button.config(command=run_train_py)
predict_button.config(command=webcam_predict)

delete_play_mp3s()
main.mainloop()
