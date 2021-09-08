# GoldenTime
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE) #yolov4-deepsort

## 개발 이유
&nbsp;판단력이 흐린 아이 또는 보행자가 차량 뒤에서 급히 뛰쳐나오는 경우, 운전자가 불법주정차로 인한 사각지대로 인해 보행자를 인식하지 못한  
경우 사고로 이어지는 상황이 적지 않음에 중점을 두고 해당 상황에 운전자, 보행자에게 경고해 주는 기능을 구현하여 사고를 예측, 방지하고자 함


## Description
&nbsp;어린이 보호구역의 CCTV와 YOLO v4를 활용하여 차량과 사람을 인식, Deep sort를 활용하여 개별 ID를 부여한다.  
사람과 차량의 충돌(접촉)이 일어나는 경우 위험 등급을 상향 시키고, 지속적인 충돌로 일정 위험 등급에 도달하였을 때 경고  
(인식 박스의 색이 빨간색으로 변하고 비프음 출력)하여 차량 뒤에서 뛰쳐나오는, 운전자가 인식하지 못한 사람의 사고율을 낮춘다.  
AABB 충돌 알고리즘을 이용하여 YOLOv4로 인식된 bounding BOX 간 충돌, 그중에서도 차량과 사람 간의 충돌만을 인식한다.  
충돌의 정확성을 올리기 위해 기본 box의 크기를 일정 확률로 줄이고, 차량 내 운전자는 충돌에서 제외하여 의미 없는 충돌을 배제한다.

## Environment
&nbsp;시작하려면 Anaconda 또는 Pip을 통해 적절한 종속성을 설치해야 한다.  
GPU를 사용하는 사람들에게는 CUDA Toolkit 버전을 구성하므로 Anaconda 경로를 권장.

### Conda (Recommended)
```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml #conda 가상환경 생성
conda activate yolov4-cpu         #conda 가상환경 실행

# Tensorflow GPU
conda env create -f conda-gpu.yml #conda 가상환경 생성
conda activate yolov4-gpu         #conda 가상환경 실행
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.) pip버전 19.0이상
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
이 저장소에서 사용하는 TensorFlow 버전에 적합한 버전인 CUDA Toolkit 버전 10.1을 사용해야 한다.  
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## YOLOv4 weights파일
트래커에 대해 사전 훈련된 YOLOv4 가중치:  
https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT  
다운 후 data폴더에 넣어준다.

## tensorflow 을 사용하기 위한 yolov4 weights 파일 변환.
&nbsp;YOLOv4를 사용하여 객체 추적을 구현하려면 먼저 save_model.py를 사용하여 .weights를 체크포인트 폴더에 저장될 해당 TensorFlow 모델로 변환해야한다.  
그 후 python goldentime_gui.py 스크립트를 실행하여 YOLOv4, DeepSort 및 TensorFlow로 객체 추적
```
python save_model.py --model yolov4 
```

## 실행 명령어.
```bash
python goldentime.py --video ./data/video/cars.mp4 --output ./outputs/demo.avi --model yolov4 check_crash

python goldentime_gui.py --video ./data/video/cars.mp4 --output ./outputs/demo.avi --model yolov4 check_crash #gui(pyqt5)
```
## 실행 영상 (goldentime_gui.py)
<p align="center"><img src="test.gif"\></p>



## Command Line Args Reference
```
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 goldentime.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
  --check_crash: check object box crash
    (default: True)
```



### References  
  fork : [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)

  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
