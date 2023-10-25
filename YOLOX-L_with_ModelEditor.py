#!/usr/bin/env python
# coding: utf-8

# Overall purpose: apply optimization options to compile and run the YOLOX-L model.

# The Furiosa SDK offers various options for optimizing the compilation and execution of models.
# This script provides an example of compiling and running the YOLOX model more optimally
# using the options provided by the Furiosa SDK.

# 1. Preparation
# To run this script, you need to install the latest version of
# essential Linux packages for the Furiosa SDK and set up a Python execution environment.
# If you haven't installed the Linux packages or configured the Python execution environment yet,
# you can prepare by referring to the following two documents:

# * [Driver, Firmware, Runtime Installation Guide](https://furiosa-ai.github.io/docs/latest/en/software/installation.html)
# * [Configuring the Python Execution Environment](https://furiosa-ai.github.io/docs/latest/en/software/python-sdk.html#python)

# Once you have prepared the essential Linux packages and the Python execution environment,
# you can proceed to install the latest version of the Furiosa SDK Python package and the
# quantization tools additional package with the following command:

# $ pip3 install --upgrade 'furiosa-sdk[quantizer]'


# Finally, you will also need the `opencv-python-headless` package, which provides
# Python bindings for OpenCV. It is used to read and preprocess image files in the examples.
# You can install it with the following command:

# $ pip3 install opencv-python-headless


# 1.1 YOLOX-L Model

# YOLOX-L 모델 구현으로는 Megvii 사가 [공개](https://yolox.readthedocs.io/en/latest/demo/onnx_readme.html)한 ONNX 모델 [yolox_l.onnx](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx)

# Use the provided model and download it to the current directory.
# 
# $ wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx

# ### 1.2 데이터 집합

# We will use the [COCO dataset](https://cocodataset.org) for calibration data and performance
# measurement, consisting of the 2017 [validation dataset](http://images.cocodataset.org/zips/val2017.zip) (1 GB)
# and the [test dataset](http://images.cocodataset.org/zips/test2017.zip) (6 GB).
# To maintain a directory structure as shown in the `tree` command output below,
# create a `coco` directory under the current directory, and then extract the downloaded dataset archive files into that `coco` directory.

# (Here, I cannot display the `tree` command output, but you should create a directory structure similar to the COCO dataset's original structure.)

# 1.1 YOLOX-L Model

# For the implementation of the YOLOX-L model, we will use the ONNX model
# [yolox_l.onnx](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0)
# that Megvii has [published](https://yolox.readthedocs.io/en/latest/demo/onnx_readme.html).
# Download the model and save it in the current directory.
# 
# $ wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx


# 1.2 Dataset

# We will use the [COCO dataset](https://cocodataset.org) for both calibration data to determine
# quantization parameters and test data to measure performance. Download the
# 2017 [validation dataset](http://images.cocodataset.org/zips/val2017.zip) (1 GB) and
# [test dataset](http://images.cocodataset.org/zips/test2017.zip) (6 GB).
# Create a directory called `coco` in the current directory to maintain the directory structure
# similar to the `tree` command output below, and unzip the downloaded dataset files into that `coco` directory.
#
# 
# $ mkdir coco
# 
# $ wget http://images.cocodataset.org/zips/val2017.zip
# $ unzip -d coco val2017.zip
# 
# $ wget http://images.cocodataset.org/zips/test2017.zip
# $ unzip -d coco test2017.zip
# 
# $ tree coco
# coco
# ├── test2017
# │   ├── 000000000001.jpg
# │   ├── 000000000016.jpg
# │   ├── 000000000019.jpg
# │   ...
# │   ├── 000000581911.jpg
# │   └── 000000581918.jpg
# └── val2017
#     ├── 000000000139.jpg
#     ├── 000000000285.jpg
#     ├── 000000000632.jpg
#     ...
#     ├── 000000581615.jpg
#     └── 000000581781.jpg
# 

# Additionally, label information corresponding to each class ID is also required.
# $ wget https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

# 1.3 imports

import glob
from itertools import islice
import time

import cv2
import numpy as np
import onnx
import torch
import torchvision
import tqdm

import furiosa.native_runtime
from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod, ModelEditor, TensorType

# 2. YOLOX-L Model
# Load the YOLOX-L model into memory.

model = onnx.load_model("models/yolox_l.onnx")

# This model has been trained using a preprocessed image dataset with the `calibrate_preproc` function.
# To provide a more specific description of what the `calibrate_preproc` preprocessing function does,
# lines 2 to 13 resize the image data to match the model's input size. The original image's aspect ratio
# is preserved while it is scaled up or down, and the empty spaces are filled with pixels having a value of 114.
# Then, at line 15, the channel (C) axis is moved to the front. In other words, it transforms the data from
# the HxWxC format to the CxHxW format. Finally, at line 16, it converts uint8 values to float32 values.
# Overall, this is a common preprocessing type used for model inputs that deal with images.

# https://github.com/Megvii-BaseDetection/YOLOX/blob/68408b4083f818f50aacc29881e6f97cd19fcef2/yolox/data/data_augment.py#L142-L158

def calibrate_preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:  # line 2
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[
        : int(img.shape[0] * r), : int(img.shape[1] * r)
    ] = resized_img  # line 13

    padded_img = padded_img.transpose(swap)  # line 15
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)  # line 16
    return padded_img, r

# 3. Calibration and Quantization

# Prepare the calibration dataset that will be used to determine the quantization parameters.
# In this example, for a quick demonstration, we will use 100 randomly selected images from the
# COCO validation dataset. Note that the MLPerf benchmark uses 500 images as the calibration dataset.
# It's important to mention that the same preprocessing (`calibrate_preproc`) used during training
# is also applied here. The `[np.newaxis, ...]` part in line 2 transforms the data from the
# shape CxHxW to the required shape of 1xCxHxW for the model input.

calibration_dataset = [
    calibrate_preproc(cv2.imread(image), (640, 640))[0][np.newaxis, ...]  # line 2
    for image in islice(glob.iglob("coco/val2017/*.jpg"), 100)
]

model = optimize_model(model)

# Perform calibration and quantization using the prepared YOLOX-L model and calibration dataset.

calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

for calibration_data in tqdm.tqdm(
    calibration_dataset, desc="Calibration", unit="images", mininterval=0.5
):
    calibrator.collect_data([[calibration_data]])

ranges = calibrator.compute_range()

# The `range` above holds calibration ranges for each tensor.
# 
# Since it stores values in dictionary format, you can save and load it in various data formats as needed.
# 
# The example below demonstrates saving and loading it in JSON format.

import json

with open("yolo_ranges.json", "w") as f:
    f.write(json.dumps(ranges, indent=4))
with open("yolo_ranges.json", "r") as f:
    ranges = json.load(f)


# 4. Compilation Optimization Options

# The Furiosa SDK provides several options that allow users to fine-tune various stages of the compilation
# process to suit the target model. One of these options is the ability to transform some commonly used #
# forms of preprocessing code in image-related models into more efficient code that can be executed in the
# Furiosa NPU environment.

# To use these options, `ModelEditor` is provided.

furiosa_editor = ModelEditor(model)

# To achieve additional performance improvements, you need to change the data types of the model's input and output.
# To change the model's input type to u8 and the output type to i8, execute the following:

# use model input tensor which name is "images" as u8 type instead of f32 type
furiosa_editor.convert_input_type("images", TensorType.UINT8)
# use model output tensor which name is "output" as i8 type instead of f32 type
furiosa_editor.convert_output_type("output", TensorType.INT8)

# Then, specify the setting `"permute_input": [[0, 2, 3, 1]]` as shown in the code below to
# efficiently transform the input from 1xHxWxC to 1xCxHxW. By providing this preprocessing
# information to the compiler, it can generate code that reduces the overall execution time #
# while producing the same computational results as the code commented out above.

compiler_config = {
    "permute_input": [
        [0, 2, 3, 1],
    ],
}

# We have used the i8 data type for the output, but this represents integer values and not the
# actual floating-point values we intend to use.

# To convert integer data to floating-point, two parameters, scale and zero point, are required.
# Scale and zero point can be determined from the minimum and maximum values of tensor elements.

# The minimum and maximum values of tensor elements are recorded in `ranges`, and you can obtain
# the minimum and maximum values for a specific tensor by using its name. To convert the i8 data
# of the output tensor to floating-point, you obtain the minimum and maximum values for the `output` tensor.

# For detailed information on quantization, you can refer to [this reference](https://gaussian37.github.io/dl-concept-quantization/).

# The `dequantize_x` function is used to convert i8 data to f32 values.

# get min max range from ranges
min_v, max_v = ranges["output"]
info = np.iinfo(np.int8)
min_q, max_q = info.min, info.max
# get scale and zeropoint
scale = (max_v - min_v) / (max_q - min_q)
zero_point = np.round((max_v * min_q - min_v * max_q) / (max_v - min_v))


# convert int8 type to f32
def deqauntize_x(x: np.ndarray) -> float:
    return (x - zero_point) * scale

# You can define the preprocessing function as follows.
# To define it, simply remove lines 15 and 16 from the `calibrate_preproc` function above.

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    #    padded_img = padded_img.transpose(swap)  # line 15
    #    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)  # line 16

    return padded_img[np.newaxis, ...]


# To interpret the model's output values, define `grids` and `strides`.

grids = []
strides = []
for (hsize, wsize), stride in zip([(80, 80), (40, 40), (20, 20)], [8, 16, 32]):
    yv, xv = np.meshgrid(  # pylint: disable=invalid-name
        np.arange(hsize), np.arange(wsize), indexing="ij"
    )
    grid = np.stack((xv, yv), 2).reshape(-1, 2)
    grids.append(grid)
    shape = grid.shape
    strides.append(np.full(shape, stride))

grids = np.concatenate(grids, axis=0, dtype=np.float32)
strides = np.concatenate(strides, axis=0, dtype=np.float32)


# Since the model's output is in i8 format, it is converted to floating-point values
# using `dequantize_x`. Based on this, the post-processing function is as follows.
# You can refer to the original model's post-processing method
# [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/419778480ab6ec0590e5d3831b3afb3b46ab2aa3/yolox/utils/boxes.py#L32-L76).

def postproc(outputs: np.ndarray, num_classes: int, conf_thre: float, nms_thre: float):
    output = [None] * len(outputs)
    for i, image_pred in enumerate(outputs):
        if image_pred.shape[0] == 0:
            continue
        class_pred = np.expand_dims(
            np.argmax(image_pred[:, 5 : 5 + num_classes], axis=1), axis=1
        )
        class_conf = deqauntize_x(
            np.take_along_axis(image_pred[:, 5 : 5 + num_classes], class_pred, axis=1)
        )

        conf_mask = (
            deqauntize_x(image_pred[:, 4]) * class_conf.squeeze() >= conf_thre
        ).squeeze()
        detections = image_pred[conf_mask][:, :5]
        if detections.shape[0] == 0:
            continue

        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        detection_grids = grids[conf_mask]
        detection_strides = strides[conf_mask]

        # dequantize detection
        detections = deqauntize_x(detections)

        detections[:, :2] = (detections[:, :2] + detection_grids) * detection_strides
        detections[:, 2:4] = np.exp(detections[:, 2:4]) * detection_strides

        # apply box
        box_corner = np.empty_like(detections)
        box_corner[:, 0] = detections[:, 0] - detections[:, 2] / 2
        box_corner[:, 1] = detections[:, 1] - detections[:, 3] / 2
        box_corner[:, 2] = detections[:, 0] + detections[:, 2] / 2
        box_corner[:, 3] = detections[:, 1] + detections[:, 3] / 2
        detections[:, :4] = box_corner[:, :4]
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate(
            (detections[:, :5], class_conf, class_pred.astype(np.float32)), axis=1
        )
        nms_out_index = torchvision.ops.batched_nms(
            torch.from_numpy(detections[:, :4]),
            torch.from_numpy(detections[:, 4] * detections[:, 5]),
            torch.from_numpy(detections[:, 6]),
            nms_thre,
        )
        output[i] = detections[nms_out_index.numpy()]
    return output


# Finally, we attempt model quantization using `ranges`.
model_quantized = quantize(model, ranges)


# 5. Inference and Latency Measuremen#t

# Create a session using the quantized model and the compile settings described earlier.
# Use the created session for inference on the test dataset. Similar to the calibration,
# for a quick demonstration, we will use 1000 randomly selected images out of a total #
# of 40,670 test dataset images. Then, measure the total time taken for inference on these 1000 images.

total_predictions = 0
elapsed_time = 0
len_of_images = 1000
list_of_image_name_and_detection = []
with furiosa.native_runtime.sync.create_runner(
    model_quantized, compiler_config=compiler_config
) as session:
    for image in tqdm.tqdm(
        islice(glob.iglob("coco/test2017/*.jpg"), len_of_images),
        desc="Evaluation",
        unit="images",
        mininterval=0.5,
        total=len_of_images,
    ):
        start = time.perf_counter_ns()
        inputs = preproc(cv2.imread(image), (640, 640))
        session_output = session.run(inputs)
        detection = postproc(session_output[0], 80, 0.25, 0.65)
        elapsed_time += time.perf_counter_ns() - start
        list_of_image_name_and_detection.append((image, detection[0]))
        total_predictions += 1


# 1000개 이미지를 추론하는데 걸린 시간으로부터 평균 레이턴시(latency)를 계산합니다. 전/후 처리를 포함하여 31.059 ms 걸리는 걸 확인할 수 있습니다. 
# 

# In[50]:


latency = elapsed_time / total_predictions
print(f"Average Latency: {latency / 1_000_000} ms")


# 6. Model Evaluation
# Let's verify if the model produces accurate results. First, determine the number of boxes you want to represent.

num_of_box = 5

# The next step is to retrieve the image to be used and the detection results.
index = [
    l.shape[0] for _, l in list_of_image_name_and_detection if l is not None
].index(num_of_box)
image_name, sample_detection = list_of_image_name_and_detection[index]

# Implement the `plot_one_box` function that draws a box on the image.

def plot_one_box(x, img, color, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


# You can obtain the label name corresponding to the object ID through the `coco-labels-2014_2017.txt` file.
with open("coco-labels-2014_2017.txt", "r") as f:
    names = f.readlines()


# Now let's proceed to draw the recognized boxes on the image.
sample_image = cv2.imread(image_name)
for each_box in sample_detection:
    xyxy = each_box[:4]
    conf = each_box[4] * each_box[5]
    cls = each_box[6]
    label = "%s %.2f" % (names[int(cls)], conf)
    plot_one_box(xyxy, sample_image, label=label, color=[0, 128, 128], line_thickness=1)


# Save the completed image to `test.jpg` and display the image.
cv2.imwrite("test.jpg", sample_image)

# 7. Conclusion

# In conclusion, we have used the Furiosa SDK to apply `ModelEditor` to the YOLOX-L model, optimizing it for better performance compared to the original model, and we have demonstrated the model's results in the form of images.
# `ModelEditor` provides an easy way to apply optimization options effectively.

