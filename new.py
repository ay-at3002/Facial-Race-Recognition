from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import cv2
import argparse
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def detect_face(image_paths, SAVE_DETECTED_AT, default_max_size=800, size=300, padding=0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            print('---%d/%d---' % (index, len(image_paths)))
        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(SAVE_DETECTED_AT, path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            dlib.save_image(image, face_name)

def predidct_age_gender_race(save_prediction_at, imgs_path='cropped_faces/'):
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/fairface_alldata_20191111.pt'))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('fair_face_models/fairface_alldata_4race_20191111.pt'))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           race_preds_fair_4,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair, race_scores_fair_4,
                           gender_scores_fair,
                           age_scores_fair, ]).T
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'race_scores_fair_4',
                      'gender_scores_fair',
                      'age_scores_fair']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # race fair 4

    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    result[['face_name_align',
            'race', 'race4',
            'gender', 'age',
            'race_scores_fair', 'race_scores_fair_4',
            'gender_scores_fair', 'age_scores_fair']].to_csv(save_prediction_at, index=False)

    print("saved results at ", save_prediction_at)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def reverse_resized_rect(rect, resize_ratio):
    # Assuming rect is a dlib rectangle
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    rect_tpl = (int(x / resize_ratio), int(y / resize_ratio), int(w / resize_ratio), int(h / resize_ratio))
    rect_in_origin = (int(x), int(y), int(w), int(h))
    return rect_tpl, rect_in_origin


def detect_and_predict_realtime(model_fair_7, model_fair_4, sp, cnn_face_detector, img, resize_ratio, device, default_max_size=800):
    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(old_height * resize_ratio)
    else:
        new_width, new_height = int(old_width * resize_ratio), default_max_size

    img_resized = dlib.resize_image(img, cols=new_width, rows=new_height)
    dets = cnn_face_detector(img_resized, 1)

    for detection in dets:
        rect = detection.rect
        rect_tpl, rect_in_origin = reverse_resized_rect(rect, resize_ratio)

        image_face = dlib.get_face_chip(img, sp(img_resized, rect))
        image_face = cv2.cvtColor(image_face, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
        image_face = cv2.resize(image_face, (224, 224))  # Resize to match the model input size
        image_face = transforms.ToTensor()(image_face)
        image_face = image_face.unsqueeze(0).to(device)

        # Perform prediction using the provided models (model_fair_7, model_fair_4)
        outputs_7 = model_fair_7(image_face)
        outputs_7 = outputs_7.cpu().detach().numpy()
        outputs_7 = np.squeeze(outputs_7)

        outputs_4 = model_fair_4(image_face)
        outputs_4 = outputs_4.cpu().detach().numpy()
        outputs_4 = np.squeeze(outputs_4)

        race_pred_7 = np.argmax(outputs_7[:7])
        race_pred_4 = np.argmax(outputs_4[:4])
        gender_pred = np.argmax(outputs_7[7:9])
        age_pred = np.argmax(outputs_7[9:18])

        # Example: Print the predicted results
        print("Predicted Race (Fair 7):", race_pred_7)
        print("Predicted Race (Fair 4):", race_pred_4)
        print("Predicted Gender:", gender_pred)
        print("Predicted Age:", age_pred)

# ... (rest of the code remains unchanged)


def main_realtime(resize_ratio=0.5, batch_size=4):
    # Load your models and other configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model for 7-race prediction
    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('/Users/ayushatray/Downloads/RaceRecognition/res34_fair_align_multi_7_20190809.pt',map_location=device))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    # Load the model for 4-race prediction
    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('/Users/ayushatray/Downloads/RaceRecognition/res34_fair_align_multi_4_20190809.pt',map_location=device))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    # Load dlib models
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')

    # ... (other configurations)

    # OpenCV setup for capturing from webcam (you can adjust the device index as needed)
    cap = cv2.VideoCapture(0)

    while True:
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            frame = cv2.resize(frame, (800, 600))
            frames.append(frame)

        # Process frames in batch
        for frame in frames:
            detect_and_predict_realtime(model_fair_7, model_fair_4, sp, cnn_face_detector, frame, resize_ratio, device)

        cv2.imshow("Real-Time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main_realtime()
