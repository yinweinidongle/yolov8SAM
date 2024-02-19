from ultralytics import YOLO
import numpy as np
import cv2
import os
import sys
import torch
from PIL import Image

import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join("/Users/yinwei/Downloads", "", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

yolo_model = YOLO(r'daofu_detect_1102.pt')

#IMAGE_NAME = r'E:\python-workspace\dataset\huantai_yolo\val\images\139.tif'
# IMAGE_NAME = r'E:\python-workspace\data2_1024\images\149.png'
# IMG_SOURCE = cv2.imread(IMAGE_NAME)
# img = cv2.cvtColor(IMG_SOURCE, cv2.COLOR_BGR2RGB)
#
# print(img.shape)


#获所有图片
import glob
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 获取当前目录
#directory = r'D:\我的文档\WeChat Files\wxid_ecl0vbxppn1p41\FileStorage\File\2023-07\huantai\1_all'
#directory = r'E:\python-workspace\data2_1024\images'
#E:\python-workspace\ultralytics-main\sam_yolo_mask
#directory = "D:\\Download\\sanshan\\daofu_no"
directory = '/Users/yinwei/Downloads/daofu_png'
#directory = r'D:\Download\sanshan\ss101601'

# 获取所有文件
files = glob.glob(directory + "/*")

yolo_model = YOLO('daofu_detect_1102.pt')

model_cls = YOLO('daofu_cls_best.pt')


# 输出所有文件名
for file in files:
    #print(file)



    # IMAGE_NAME = r'E:\python-workspace\dataset\huantai_yolo\val\images\139.tif'
    IMAGE_NAME = file
    IMG_SOURCE = cv2.imread(IMAGE_NAME)
    #cv2.imshow("cccc", img)
    #img = cv2.cvtColor(IMG_SOURCE, cv2.COLOR_BGR2RGB)
    img = IMG_SOURCE

    #cv2.imshow("dddd",img)

    results = yolo_model.predict(source=IMG_SOURCE)


    if(len(results[0].boxes)>0):


        boxes_class_name = np.array([])
        for r in results:
            for c in r.boxes.cls:
                boxes_class_name = np.append(boxes_class_name,yolo_model.names[int(c)])

        k = 0
        result_np_son = np.array([])
        result_class = np.array([])
        result_score = np.array([])
        for result in results:
            result_np = np.array(result.boxes.data.cpu())

        for i in result_np:
            result_np_son = np.append(result_np_son,i[:][:-2])
            result_class = np.append(result_class,i[:][-1:])
            result_score = np.append(result_score,i[:][-2:-1])
            k += 1


        boxes = result_np_son.reshape((k,4))
        boxes_class = result_class.reshape((k,))
        boxes_score = result_score.reshape((k,))

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        import supervision as sv

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

        mask_predictor = SamPredictor(sam)



        segmented_image = img.copy()
        box_image = img.copy()
        boxes_class_value = {'0':sv.Color.red(),'1':sv.Color.blue(),'2':sv.Color.red(),'3':sv.Color.blue() ,'4':sv.Color.black(),'5':sv.Color.green()}
        mask_predictor.set_image(img)

        boxes = result.boxes.xyxy

        masks, scores, logits = mask_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes * 1040 / max(img.shape),
            multimask_output=False
        )

        masks = masks.cpu().numpy()

        filename0 = os.path.basename(IMAGE_NAME)

        index_mask = 0





        maskTmp = np.zeros((1024,1024))

        i = 0
        for box in boxes.cpu().numpy():
            box_annotator = sv.BoxAnnotator(color=boxes_class_value[str(int(boxes_class[i]))], text_padding=10)
            mask_annotator = sv.MaskAnnotator(color=boxes_class_value[str(int(boxes_class[i]))])

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks[i]),
                mask=masks[i]
            )

            mask_seg = masks[i][0]
            mask_seg = torch.tensor(mask_seg, dtype=torch.uint8)
            mask_seg = mask_seg.cpu().numpy()
            thresholdValue1, two_value1 = cv2.threshold(mask_seg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            masked = cv2.add(IMG_SOURCE, np.zeros(np.shape(IMG_SOURCE), dtype=np.uint8), mask=two_value1)
            filename1 = "daofu_clip/ms_" + str(index_mask) + "_{}".format(filename0)
            cv2.imwrite(filename1, masked)
            # Predict with the model
            results = model_cls(filename1)  # predict on an image

            if results[0].probs.data[0].float() <= 0.6:
                print(filename1 + "不是，置信度：" + str(results[0].probs.data[0]))
            else:
                print(filename1 + "是倒伏，置信度：" + str(results[0].probs.data[0]))
                #mask_erode = cv2.erode(masks[i][0], np.ones((3, 3), np.uint8), iterations=1)
                maskTmp += masks[i][0]
            index_mask = index_mask + 1

            # mask00 = masks[i]
            #
            # ss = os.path.basename(IMAGE_NAME) + ""
            #cv2.imwrite('final_3/mask_'+str(i)+ss, masks[i][0]*255)

            detections = detections[detections.area == np.max(detections.area)]
            box_image = box_annotator.annotate(scene=box_image, detections=detections, skip_label=True)
            segmented_image = mask_annotator.annotate(scene=segmented_image, detections=detections)



            i += 1
        # sv.plot_images_grid(
        #     images=[img, box_image, segmented_image],
        #     grid_size=(1, 3),
        #     titles=['source image', 'box image', 'segmented image']
        # )



        # 合并图像
        import random

        merged_img = cv2.addWeighted(box_image, 0.5, segmented_image, 0.5, 0)

        # 保存新的图像

        ss = os.path.basename(IMAGE_NAME) + ""
        #print(ss)
        #cv2.imwrite('final_2/'+ ss, merged_img)
        #cv2.imwrite('final_1/'+ ss, merged_img)
        cv2.imwrite('final_4/'+ ss, merged_img)

        # 保存新的mask

        ss_str = ss.split(".")[0]


        # cv2.imwrite('final_mask/mask_' + ss_str + ".png", maskTmp * 255)




        cv2.imwrite('final_mask_2/' + ss_str + ".png", maskTmp * 255)

    else:
        ss = os.path.basename(IMAGE_NAME) + ""
        ss_str = ss.split(".")[0]
        mask000 = np.zeros(IMG_SOURCE.shape[:2],dtype="uint8")
        # cv2.imwrite('final_mask/mask_' + ss_str + ".png", mask000)
        cv2.imwrite('final_mask_2/' + ss_str + ".png", mask000)
