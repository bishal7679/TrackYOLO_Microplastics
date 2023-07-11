import yolov5
#from vidgear.gears import CamGear
import cv2
import numpy as np

#stream = CamGear(source='https://www.youtube.com/watch?v=N5VJWfXoDiM', stream_mode = True, logging=True).start() # YouTube Video URL as input
stream = cv2.VideoCapture(2)
model = yolov5.load('../model/new_cameras.pt')

# infinite loop

while True:
    
    frame = stream.read()

    # print(type(img))
    results = model(frame)
    model.conf = 0.05
    results = model(frame, size=1920)

    results = model(frame, augment=True)
    label_id_offset = 1

    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, x2, y1, y2
    scores = predictions[:, 4].tolist()
    categories = predictions[:, 5].tolist()
    cv2.imshow("Detect", frame)

    print(scores)
    print(boxes)
    for category_id in categories:
        print(model.names[int(category_id)])


    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()
    # print(results.pred[0].tolist())