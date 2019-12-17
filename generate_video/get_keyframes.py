# coding=utf-8

import os
import cv2
from glob import glob
videos_src_path = "../demo/src"
frames_save_path = "../demo/src/frames/"
os.system("mkdir ../demo/src/frames/")

def get_keyframe(frame_save_path):
    cmd='cp '+frame_save_path+'000000.jpg'+' ../demo/transfer_data/'
    os.system(cmd)

def video2frame(cap, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")

    while(success):
        success, frame = cap.read()
        print ("---> 正在读取第%d帧:" % frame_index, success)

        if frame_index % interval == 0 and success:
            resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            cv2.imwrite(frames_save_path + "%06d.jpg" % frame_count, resize_frame)
            frame_count += 1

        frame_index += 1

    cap.release()


if __name__ == '__main__':
    vid_paths = sorted(glob(videos_src_path + '/*.mp4'))
    for vid_path in vid_paths:
        cap = cv2.VideoCapture(vid_path)
        video2frame(cap, frames_save_path, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1)
    get_keyframe(frames_save_path)