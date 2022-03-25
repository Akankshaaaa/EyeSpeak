from time import time
import cv2
import dlib
import time

from cv2 import WINDOW_NORMAL
from imutils import face_utils
from utils import *
import sys

keyboard = np.zeros((400, 800, 3), np.uint8)

# DEFINE KEYBOARD KEYS
keys = [["1", "Q", "A", "Z", "_"],
        ["2", "W", "S", "X", "_"],
        ["3", "E", "D", "C", "_"],
        ["4", "R", "F", "V", "_"],
        ["5", "T", "G", "B", "_"],
        ["6", "Y", "H", "N", "_"],
        ["7", "U", "J", "M", "_"],
        ["8", "I", "K", ",", "_"],
        ["9", "O", "L", ".", "_"],
        ["0", "P", "<-", "?", "_"]]

size_blink = (34, 26)
size_gaze = (64, 56)

blink = Blink_detection()

# user defined class for gaze detection
gaze = Gaze_detection()
gaze_labels = ['center', 'left', 'right']

white_board = np.ones((80, 800, 3), np.uint8)


# DRAWS KEYS
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(keyboard, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(keyboard, button.text, (x + 15, y + 35),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img


def select_col(img, current_col):
    for button in current_col:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(keyboard, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
        cv2.putText(keyboard, button.text, (x + 15, y + 35),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img


def select_row(img, current_col, row):
    button = current_col[row]
    x, y = button.pos
    w, h = button.size
    cv2.rectangle(keyboard, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
    cv2.putText(keyboard, button.text, (x + 15, y + 35),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img


def pause(secs):
    init_time = time()
    while time() < init_time + secs: pass


class Button():
    def __init__(self, pos, text, size=[50, 50]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([i * 70, 70 * j], key))

cols = [buttonList[i:i + 5] for i in range(0, len(buttonList), 5)]

is_col_selected = False

typed_text = ""

# some counters
blink_count = 0
row_blink_count = 0
right_gaze_count = 0
left_gaze_count = 0
col = 0
row = 0

#######################
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("FILES/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
cap.set(3, 1200)  # width
cap.set(4, 720)  # height

while True:
    main_window = np.zeros((880, 1000, 3), np.uint8)
    cap.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = cap.read()

    _, frame = cap.read()  # reading the frame form the webcam
    frame = cv2.flip(frame, flipCode=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # coverting the bgr frame to gray scale

    cv2.putText(main_window, "INSTRUCTIONS", (600, 220), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    success, img = cap.read()
    current_col = cols[col]

    img = drawAll(img, buttonList)
    img = select_col(img, current_col)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # coverting the bgr frame to gray scale
    faces = detector(gray)  # this returns the dlib rectangle

    # extracting the rectangle which contain the upper and lower coordinates of the face
    if len(faces) == 0:
        continue
    face = faces[0]
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    # EYE CROPING FOR BLINK
    eye_img_l, eye_rect_l = blink.crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = blink.crop_eye(gray, eye_points=shapes[42:48])

    # EYE CROPING FOR GAZE
    eye_img_l_r = gaze.crop_eye(gray, eye_points=shapes[36:42])

    eye_img_l = cv2.resize(eye_img_l, dsize=size_blink)
    eye_img_r = cv2.resize(eye_img_r, dsize=size_blink)
    # eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    # RESIZING THE CROPPED GAZE INPUT EYE
    eye_img_l_r = cv2.resize(eye_img_l_r, dsize=size_gaze)

    # NORMALIZE THE EYE IMAGES
    eye_input_l = eye_img_l.copy().reshape((1, size_blink[1], size_blink[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, size_blink[1], size_blink[0], 1)).astype(np.float32) / 255.
    eye_input_l_r = eye_img_l_r.copy().reshape((1, size_gaze[1], size_gaze[0], 1)).astype(np.float32) / 255.

    cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(64, 224, 208), thickness=2)
    cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 0, 0), thickness=2)

    # detect blink
    left_blink, right_blink = blink.model_predict(eye_input_l, eye_input_r)

    # detect gaze
    gaze_detect = gaze.model_predict(eye_input_l_r)

    # for selecting column
    if is_col_selected == False:
        cv2.putText(main_window, "Look Left or Right to change columns", (600, 240), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 2)
        cv2.putText(main_window, "BLINK to select column", (600, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

        if gaze_detect == 'left' and left_blink >= 0.3 and right_blink >= 0.3:
            print("left gaze")
            cv2.putText(main_window, "LEFT GAZE", (400, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            left_gaze_count = left_gaze_count + 1
            print(left_gaze_count)
        if gaze_detect == 'center' and left_blink >= 0.3 and right_blink >= 0.3:
            print("center gaze")
            cv2.putText(main_window, "CENTER GAZE", (400, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        if gaze_detect == 'right' and left_blink >= 0.3 and right_blink >= 0.3:
            print("right gaze")
            cv2.putText(main_window, "RIGHT GAZE", (400, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            right_gaze_count = right_gaze_count + 1

        print(left_blink, right_blink)
        if left_blink < 0.3 and right_blink < 0.3:
            print("blink detected")
            cv2.putText(main_window, "--BLINK--", (400, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            blink_count = blink_count + 1

        # IF BLINKED FOR 3 SECONDS THEN SELECT THE COLUMN
        if blink_count == 3:
            is_col_selected = True
            print(" voluntary blink detected")
            sys.stdout.write('\a')
            sys.stdout.flush()
            select_row(img, current_col, row)
            row = 0
            blink_count = 0

        # IF LOOKING LEFT FOR 3 OR MORE FRAMES THEN MOVE LEFT
        if left_gaze_count >= 3:
            left_gaze_count = 0
            print("voluntary left gaze detected")
            sys.stdout.write('\a')
            sys.stdout.flush()
            if col > 0:
                col = col - 1

            else:
                col = 9

        # IF LOOKING RIGHT FOR 3 OR MORE FRAMES THEN MOVE RIGHT
        if right_gaze_count >= 3:
            right_gaze_count = 0
            print("voluntary right gaze detected")
            sys.stdout.write('\a')
            sys.stdout.flush()
            if col < 9:
                col = col + 1
            else:
                col = 0

    # for selecting row
    else:
        cv2.putText(main_window, "Look LEFT to go UP", (600, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.putText(main_window, "Look RIGHT to go DOWN", (600, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.putText(main_window, "BLINK to TYPE the key", (600, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.putText(main_window, "BLINK LEFT EYE to EXIT", (600, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

        print(col, row)
        text = keys[col][row]
        select_row(img, current_col, row)

        if gaze_detect == 'left' and left_blink >= 0.3 and right_blink >= 0.3:
            print("left gaze")
            left_gaze_count = left_gaze_count + 1
            print(left_gaze_count)
        if gaze_detect == 'center' and left_blink >= 0.3 and right_blink >= 0.3:
            print("center gaze")
        if gaze_detect == 'right' and left_blink >= 0.3 and right_blink >= 0.3:
            print("right gaze")
            right_gaze_count = right_gaze_count + 1
            print(right_gaze_count)

        # LOOK LEFT TO MOVE UP
        if left_gaze_count >= 3:
            left_gaze_count = 0
            print(text)
            cv2.putText(main_window, "UP", (440, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            sys.stdout.write('\a')
            sys.stdout.flush()
            if row > 0:
                row = row - 1
            else:
                row = 4

        # LOOK RIGHT TO MOVE DOWN
        elif right_gaze_count >= 3:
            right_gaze_count = 0
            print(text)
            cv2.putText(main_window, "DOWN", (440, 380), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            sys.stdout.write('\a')
            sys.stdout.flush()
            if row == 4:
                row = 0
            else:
                row = row + 1

        # BLINK WITH LEFT EYE TO EXIT FROM ROW SELECTION
        if left_blink < 0.3 and right_blink > 0.3:
            is_col_selected = False

        if left_blink < 0.3 and right_blink < 0.3:
            row_blink_count = row_blink_count + 1

        # BLINK TO SELECT THE LETTER
        if row_blink_count == 2:
            row_blink_count = 0
            is_col_selected = False
            print("typed text:", text)
            if text == "_":
                typed_text = typed_text + " "
            elif text == "<-":
                typed_text = typed_text[:-1]
            else:
                typed_text = typed_text + text

            # TYPE THE SELECTED LETTER
            white_board[:] = (0, 0, 0)
            sys.stdout.write('\a')
            sys.stdout.flush()
            cv2.putText(white_board, typed_text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Combining all windows into single window:
    main_window[50:150, 600:700] = cv2.resize(cv2.cvtColor(eye_img_l, cv2.COLOR_BGR2RGB), (100, 100))
    cv2.putText(main_window, "LEFT EYE", (600, 170), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 51), 2)
    main_window[50:150, 750:850] = cv2.resize(cv2.cvtColor(eye_img_r, cv2.COLOR_BGR2RGB), (100, 100))
    cv2.putText(main_window, "RIGHT EYE", (750, 170), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 51), 2)

    main_window[400:800, 150:950] = keyboard
    main_window[50:350, 150:550] = cv2.resize(frame, (400, 300))
    main_window[790:870, 100:900] = white_board

    cv2.namedWindow("EyeSpeak", WINDOW_NORMAL)

    cv2.imshow("EyeSpeak", main_window)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
