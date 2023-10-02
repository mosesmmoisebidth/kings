import cv2 as cv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import os
import IPython
import subprocess
from IPython.display import Video, display


path = 'https://docs.google.com/uc?export=download&confirm=&id=1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-'

frac = 0.65
display(Video(data=path, height=int(frac*720), width=int(frac*1280)))
model = YOLO("yolov8x.pt")
dict_classnames = model.model.names
print("the dict_classnames are: {}".format(dict_classnames))

def resize_frame(frame, scale_percent):
    width = (frame.shape[1] * scale_percent) / 100
    height = (frame.shape[0] * scale_percent) / 100
    dim = (width, height)
    resize = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    return resize

scale_percent = 50
verbose = False
video_path = ""

cap = cv.VideoCapture(video_path)
class_IDS = [2, 3, 5, 7]
centers_old = {}
centers_new = {}
obj_id = 0
vehicle_enter_in = dict.fromkeys(class_IDS, 0)
vehicle_enter_out = dict.fromkeys(class_IDS, 0)
end = []
frames_list = []
cy_linha = int(1500 * scale_percent/100)
cx_sentido = int(2000 * scale_percent/100)
offset = int(8 * scale_percent/100)
vehicle_in_count = 0
vehicle_out_count = 0

print("verbose during prediction: {}".format(verbose))

height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv.CAP_PROP_FPS)
print("the original frame is: {}".format((width, height)))

if scale_percent != 100:
    print("INFO: Scaling may change due to the pixel lines change")
    width = int((width * scale_percent) / 100)
    height = int((height * scale_percent) / 100)
    print("INFP: the ne dim: {}".format((width, height)))

video_name = ""
output_path = ""
VIDEO_CODEC = "MP4V"
output_video = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))

for i in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
    ret, frame = cap.read()
    if not ret:
        print("INFO: The video specified cannot be read convert to another format may be please")
    frame = resize_frame(frame, scale_percent)
    if verbose:
        print("INFO: the frame was resized: {}".format(frame.shape[1], frame.shape[0]))
    y_hat = model.predict(frame, conf=0.7, classes=class_IDS, device='cpu', verbose=False)
    boxes = y_hat[0].boxes.xyxy.cpu.numpy()
    conf = y_hat[0].boxes.conf.cpu.numpy()
    classes = y_hat[0].boxes.classes.cpu.numpy()

    positions_frame = pd.DataFrame(y_hat[0].cpu.numpy().boxes.boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'classes'])
    labels = [dict_classnames for i in classes]
    cv.line(frame, (0, cy_linha), (int(4500 * scale_percent/100), cy_linha), (255, 0, 0), 8)

    for ix, rows in enumerate(positions_frame.iterrows()):
        xmin, ymin, xmax, ymax, confidence, classes = rows[1].astype('int')
        center_x, center_y = int(((xmin + xmax)/2)), int(((ymin + ymax)/2))
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv.putText(img=frame, text=labels[ix]+'-'+str(np.round(conf[ix], 2)), org=(xmin, ymin-10), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
            if (center_x >= 0) and (center_x < cx_sentido):
                vehicle_in_count += 1
                vehicle_enter_in[category] += 1

            else:
                vehicle_out_count += 1
                vehicle_enter_out[category] += 1

    contador_in_plt = [f'{dict_classnames[k]}: {i}' for k, i in vehicle_enter_in.items()]
    contador_out_plt = [f'{dict_classnames[k]}: {i}' for k, i in vehicle_enter_out.items()]

    # drawing the number of vehicles in\out
    cv.putText(img=frame, text='N. vehicles In',
                org=(30, 30), fontFace=cv.FONT_HERSHEY_TRIPLEX,
                fontScale=1, color=(255, 255, 0), thickness=1)

    cv.putText(img=frame, text='N. vehicles Out',
                org=(int(2800 * scale_percent / 100), 30),
                fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0), thickness=1)

    # drawing the counting of type of vehicles in the corners of frame
    xt = 40
    for txt in range(len(contador_in_plt)):
        xt += 30
        cv.putText(img=frame, text=contador_in_plt[txt],
                    org=(30, xt), fontFace=cv.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0), thickness=1)

        cv.putText(img=frame, text=contador_out_plt[txt],
                    org=(int(2800 * scale_percent / 100), xt), fontFace=cv.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0), thickness=1)

    # drawing the number of vehicles in\out
    cv.putText(img=frame, text=f'In:{vehicle_in_count}',
                org=(int(1820 * scale_percent / 100), cy_linha + 60),
                fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0), thickness=2)

    cv.putText(img=frame, text=f'Out:{vehicle_enter_in}',
                org=(int(1800 * scale_percent / 100), cy_linha - 40),
                fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0), thickness=2)

    if verbose:
        print(vehicle_enter_in, vehicle_enter_out)
    # Saving frames in a list
    frames_list.append(frame)
    # saving transformed frames in a output video formaat
    output_video.write(frame)

# Releasing the video
output_video.release()

####  pos processing
# Fixing video output codec to run in the notebook\browser
if os.path.exists(output_path):
    os.remove(output_path)

subprocess.run(
    ["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast", "-hide_banner", "-loglevel", "error",
     "-vcodec", "libx264", output_path])
os.remove(tmp_output_path)


for i in [28, 29, 32, 40, 42, 50, 58]:
    plt.figure(figsize =( 14, 10))
    plt.imshow(frames_list[i])
    plt.show()


