# limit the number of cpus used by high performance libraries
#22cm
from glob import glob
import os
from re import T

from importlib_metadata import NullFinder
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data = []

first_time = 0
first_time_cross=0

detection_first_time = False
detection_cross_time = False
count_save_path=""

no_detection_start_time = time_sync()

poly_line = np.array([[0,0]], np.int32)
poly_line_dict = {}

#velocity variables
velocity_time_dic = {}
velocity_line_distance = 100
velocity_actual_distnace = 2.54

def calculate_framerate(frame_rate_calc,t1,freq):
    print('FPS: {0:.2f}'.format(frame_rate_calc))
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    return frame_rate_calc

#freq = cv2.getTickFrequency()
# inside while loop   t1 = cv2.getTickCount()
# frame_rate_calc = calculate_framerate(frame_rate_calc,t1,freq)

def detect(opt):
    
    global poly_line,count_save_path,poly_line_dict,no_detection_start_time
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    #webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')


    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'
    count_save_path = str(Path(save_dir)) + '/count_' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):

        # remove polyline dict if no item in frame for 5 seconds
        if time_sync()-no_detection_start_time > 5:
            poly_line_dict={}

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    no_detection_start_time = time_sync()
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        
                        bboxes = output[0:4]
                        #print(outputs)
                        id = output[4]
                        cls = output[5]
                        #count
                        v = count_and_calculate_velocity(bboxes,w,h,id,t1)
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f} V:{v}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        #if id == 22:
                        cntr= (int(bboxes[0]+(bboxes[2]-bboxes[0])/2) , int(bboxes[1]+(bboxes[3]-bboxes[1])/2))
                        poly_line = np.append(poly_line, [cntr],axis=0)
                        
                        #poly_line_dict[id].append(np.append())
                        
                        #if len(poly_line) == 2:
                        #    poly_line = np.delete(poly_line, (0), axis=0)

                        if id not in poly_line_dict:
                            poly_line_dict[id] = np.array([cntr], np.int32)
                        else:
                            poly_line_dict[id] = np.append(poly_line_dict[id], [cntr],axis=0)
                            
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            print("text path",txt_path)
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                fpsm = 1 / (t3 - t2)
                LOGGER.info(f'FPS: {fpsm:.1f}')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                global count
                color=(0,255,0)
                
                
                #start_point = (0, h-350)
                #end_point = (w, h-350)
                 
                #velocity calculation line 
                v_start_point = (int(w/2)-velocity_line_distance, 200)
                v_end_point = (int(w/2)-velocity_line_distance, h-200)

                cv2.line(im0, v_start_point, v_end_point, color, thickness=2) 

                #middle line
                start_point = (int(w/2), 0)
                end_point = (int(w/2), h)
                cv2.line(im0, start_point, end_point, color, thickness=2)
    
                
                #if len(poly_line) > 0:
                #    cv2.polylines(im0, [poly_line], False, (140,120,255),1)


                for key, value in poly_line_dict.items():
                    if len(poly_line_dict[key]) > 0:
                        cv2.polylines(im0, [poly_line_dict[key]], False, (140,120,255),1)


                thickness = 3
                #org = (150, 150)
                org = (50, 70)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                cv2.putText(im0, str(count), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

def save_data_on_file(id,velocity):
    global count_save_path,count
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(count_save_path, 'a') as f:
        f.write(('%s ' * 4 + '\n') % (now,count,id,velocity))
        # f.write(('%s ' * 2 + '\n') % (now,count))


def count_and_calculate_velocity(box,w,h,id,t1):
    velocity = 0
    global count,data,first_time,poly_line,first_time,first_time_cross
    global detection_first_time,detection_cross_time,velocity_time_dic,velocity_actual_distnace,velocity_line_distance
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))

    if first_time == 0:
        first_time = 1
        detection_first_time = datetime.now()

    center_coordinate_x = center_coordinates[0]

    if center_coordinate_x > (int(w/2) - velocity_line_distance) and center_coordinate_x < int(w/2):
        #check key not in dict
        if id not in velocity_time_dic:
            velocity_time_dic[id] = t1

    if center_coordinate_x > int(w/2):
        if first_time_cross == 0:
            first_time_cross = 1
            detection_cross_time = datetime.now()
            diff = detection_cross_time - detection_first_time
            print("time required = ",diff.seconds)
            
        if  id not in data:
            count += 1
            data.append(id)
            

            if id in velocity_time_dic:
                time_diff = t1 - velocity_time_dic[id]
                velocity = round(velocity_actual_distnace/time_diff,2)
                print("velocity = ",velocity)
          
            save_data_on_file(id,velocity)

        #calculate velocity
            

    return velocity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--yolo_model', nargs='+', type=str, default='model/best_1.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/Traffic.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
