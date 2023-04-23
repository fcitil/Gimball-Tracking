import argparse
import logging
import os
import platform
import sys
import time
from datetime import datetime, timedelta
from os.path import dirname, join, realpath
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import serial
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import (
    LoadImages, LoadStreams)
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, SETTINGS,
                                           callbacks, colorstr, ops)
from yolov8.ultralytics.yolo.utils.checks import (check_file, check_imgsz,
                                                  check_imshow,
                                                  check_requirements,
                                                  print_args)
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.ops import (Profile, non_max_suppression,
                                               process_mask,
                                               process_mask_native,
                                               scale_boxes)
from yolov8.ultralytics.yolo.utils.plotting import (Annotator, colors,
                                                    save_one_box)
from yolov8.ultralytics.yolo.utils.torch_utils import select_device

from libs.objcenter import ObjCenter
from libs.pid import PID
from libs.servocontrol import send_data, start_link
from trackers.multi_tracker_zoo import create_tracker

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

objX = 0
objY = 0
centerX = 1920//4
centerY = 320
outputX = 0
outputY = 0
found = False
found_state = False
# limit the number of cpus used by high performance libraries
found_states = [False, False, False]


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    # add strong_sort ROOT to PATH
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def found_state_selector():
    global found_state, found_states
    while True:
        found_states.pop(0)
        found_states.append(found)
        if found_states.count(True) > 1:
            found_state = True
        else:
            found_state = False
        # print(found_states)
        time.sleep(0.05)


def pid_processX(p, i, d):

    global outputX

    p = PID(p, i, d)
    p.initialize()

    while True:
        error = centerX - objX
        # print(centerCoord,objCoord,error)
        outputX = p.update(error, found=found_state)
        time.sleep(0.01)


def pid_processY(p, i, d):

    global outputY
    p = PID(p, i, d)
    p.initialize()

    while True:

        error = centerY - objY
        # print(centerCoord,objCoord,error)
        outputY = p.update(error, found=found_state)
        time.sleep(0.01)


def send_angl(ser):
    # start the serial port
    # time.sleep(5)
    while True:
        try:
            tilt = int(outputY+1200)
            pan = int(outputX+1500)
            send_data(ser, pan, tilt)
        except:
            pass


def plotter():
    time.sleep(2)
    plt.style.use('seaborn-pastel')
    x_data, y_data, time_data = [], [], []

    figure = plt.figure()
    # plt.axes(xlim=(datetime.now(), datetime.now() + timedelta(seconds=10)), ylim=(1000, 2000))
    plt.title("pid output")
    plt.xlabel("time")
    plt.ylabel("output ms for servo")
    linex, = plt.plot_date(time_data, x_data, '-')
    liney, = plt.plot_date(time_data, y_data, '-')
    # plt.axes().set_ylim(0, 2500)
    # limit y axis to 0-2500 for the matplotlib animation
    plt.ylim(0, 2500)
    # plt.axes().set_xlim(datetime.now(), datetime.now() + timedelta(seconds=10))

    def update(frame):
        x_data.append(outputX+1500)
        y_data.append(outputY+1200)
        time_data.append(datetime.now())
        if len(x_data) > 50:
            x_data.pop(0)
            y_data.pop(0)
            time_data.pop(0)
        liney.set_data(time_data, y_data)
        linex.set_data(time_data, x_data)
        figure.gca().relim()
        figure.gca().autoscale_view()
        return linex, liney

    anim = FuncAnimation(figure, update, interval=200, save_count=50)
    plt.show()

    def update(frame):
        x_data.append(outputX+1500)
        y_data.append(outputY+1500)
        time_data.append(datetime.now())
        if len(x_data) > 50:
            x_data.pop(0)
            y_data.pop(0)
            time_data.pop(0)
        liney.set_data(time_data, y_data)
        linex.set_data(time_data, x_data)
        figure.gca().relim()
        figure.gca().autoscale_view()
        return linex, liney

    anim = FuncAnimation(figure, update, interval=200, save_count=50)
    plt.show()


@torch.no_grad()
def run(
        source='3',
        yolo_weights=WEIGHTS / 'yolov8s.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=ROOT / 'trackers' / "strongsort" / \
    'configs' / ("strongsort" + '.yaml'),
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.40,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=True,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=True,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith(
        '.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    # single models after --yolo_weights
    elif type(yolo_weights) is list and len(yolo_weights) == 1:
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(
            tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(
            save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(
                    preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(
                    preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    # im.jpg, vid.mp4, ...
                    save_path = str(save_dir / p.parent.name)
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))

            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                # camera motion compensation
                if prev_frames[i] is not None and curr_frames[i] is not None:
                    tracker_list[i].tracker.camera_update(
                        prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        # rescale boxes to im0 size
                        det[:, :4] = scale_boxes(
                            im.shape[2:], det[:, :4], shape).round()
                        masks.append(process_mask_native(
                            proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(
                            proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        # rescale boxes to im0 size
                        det[:, :4] = scale_boxes(
                            im.shape[2:], det[:, :4], shape).round()
                else:
                    # rescale boxes to im0 size
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                # pass detections to strongsort

                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                # draw boxes for visualization
                if len(outputs[i]) > 0:

                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )

                    for j, (output) in enumerate(outputs[i]):
                        global bbox, id, objY, objX
                        bbox = output[0:4]
                        objX = (bbox[0] + bbox[2]) / 2
                        objY = (bbox[1] + bbox[3]) / 2
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else
                                                              (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
            else:
                pass  # tracker_list[i].tracker.pred_n_update_all_tracks()

            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    # allow window resize (Linux)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                cv2.circle(im0, (centerX, centerY), 5, (0, 0, 255), -1)

                if save_crop:
                    time_now = time.time()
                    try:
                        if time_now - time_old >= 5:
                            print(save_dir)
                            LOGGER.info(save_dir)
                            save_one_box(np.array((0, 0, 960, 640), dtype=np.int16), im0, file=save_dir / str(
                                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), BGR=True)
                            time_old = time_now
                    except:
                        time_old = time_now
                        pass

                # if len(det):
                #     cv2.circle(im0, (objX, objY), 5, (0, 255, 255), -1)

                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        # release previous video writer
                        vid_writer[i].release()
                    if vid_cap:  # vi
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    # force *.mp4 suffix on results videos
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

        # Print total time (preprocessing + inference + NMS + tracking)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
        global found
        if len(det):
            found = True
        else:
            found = False

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-proocess, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # update model (to fix SourceChangeWarning)
        strip_optimizer(yolo_weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path,
                        default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path,
                        default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true',
                        help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' /
                        'track', help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False,
                        action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False,
                        action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true',
                        help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / \
        'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt',
                       exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    # tracking thread
    tracking_thread = Thread(target=run)
    tracking_thread.daemon = True
    tracking_thread.start()

    state_theread = Thread(target=found_state_selector)
    state_theread.daemon = True
    state_theread.start()
    # pid thread
    coef = 0.03
    pidy_thread = Thread(target=pid_processY, args=(
        0.65*coef, 0.0*coef, 0.02*coef))
    pidx_thread = Thread(target=pid_processX, args=(
        0.8*coef, 0.0*coef, 0.005*coef))
    pidx_thread.daemon = True
    pidy_thread.daemon = True

    pidy_thread.start()
    pidx_thread.start()

    plotter_thread = Thread(target=plotter)
    plotter_thread.daemon = True
    plotter_thread.start()

    try:
        ser = start_link("/dev/ttyUSB0")
    except:
        ser = start_link("/dev/ttyUSB1")

    command_thread = Thread(target=send_angl, args=[ser])
    command_thread.daemon = True

    command_thread.start()

    while True:
        try:
            # print date and time  with "-" as separator
            # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(found_state)
            # print(bbox) if found else print("No bbox yet")

        except:
            print("No bbox yet")
            pass
        time.sleep(1)
