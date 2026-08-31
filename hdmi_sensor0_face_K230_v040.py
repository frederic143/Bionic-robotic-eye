from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d

import os, sys, gc, utime
import nncase_runtime as nn
import ulab.numpy as np
import aidemo

from media.media import *
from machine import FPIOA, PWM
from media.sensor import *

# =========================
# 舵机参数
# =========================
PAN_MIN  = 70
PAN_MAX  = 110
TILT_MIN = 70
TILT_MAX = 100


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


# =========================
# 舵机控制
# =========================
def servo_init():
    fpioa = FPIOA()


    fpioa.set_function(46, FPIOA.PWM2)   # 水平
    fpioa.set_function(47, FPIOA.PWM3)   # 垂直

    pwm_x = PWM(2)
    pwm_y = PWM(3)

    pwm_x.freq(50)
    pwm_y.freq(50)

    # K230 CanMV 0.4.0 不支持 PWM.enable()
    # 使用 freq() 和 duty() 直接输出 PWM
    pwm_x.duty(7.5)
    pwm_y.duty(7.5)

    return pwm_x, pwm_y


def set_servo_angle(pwm, angle):
    angle = clamp(angle, 0, 180)
    # 百分比写法
    duty = (0.5 + angle / 180 * 2.0) / 20 * 100
    pwm.duty(duty)


# =========================
# 人脸检测类
# =========================
class FaceDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.5, nms_threshold=0.2,
                 rgb888p_size=[1920,1080], display_size=[800,480], debug_mode=0):

        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]

        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8
        )

    def config_preprocess(self):
        top, bottom, left, right = self.get_padding_param()
        self.ai2d.pad([0,0,0,0, top,bottom,left,right], 0, [104,117,123])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build(
            [1,3,self.rgb888p_size[1],self.rgb888p_size[0]],
            [1,3,self.model_input_size[1],self.model_input_size[0]]
        )

    def postprocess(self, results):
        ret = aidemo.face_det_post_process(
            self.confidence_threshold,
            self.nms_threshold,
            self.model_input_size[1],
            self.anchors,
            self.rgb888p_size,
            results
        )
        return ret[0] if ret else []

    def draw_result(self, pl, dets):
        pl.osd_img.clear()
        for det in dets:
            x,y,w,h = map(int, det[:4])
            x = x * self.display_size[0] // self.rgb888p_size[0]
            y = y * self.display_size[1] // self.rgb888p_size[1]
            w = w * self.display_size[0] // self.rgb888p_size[0]
            h = h * self.display_size[1] // self.rgb888p_size[1]
            pl.osd_img.draw_rectangle(x,y,w,h,(255,255,0,255),2)

    def get_largest_face(self, dets):
        if not dets:
            return None
        return max(dets, key=lambda d: d[2] * d[3])

    def get_padding_param(self):
        dst_w, dst_h = self.model_input_size
        ratio = min(dst_w/self.rgb888p_size[0], dst_h/self.rgb888p_size[1])
        new_w = int(self.rgb888p_size[0]*ratio)
        new_h = int(self.rgb888p_size[1]*ratio)
        return 0, dst_h-new_h, 0, dst_w-new_w


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    rgb888p_size = [1920,1080]
    display_size = [1920,1080]

    kmodel_path  = "/sdcard/examples/kmodel/face_detection_320.kmodel"
    anchors_path = "/sdcard/examples/utils/prior_data_320.bin"

    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((4200,4))

    pl = PipeLine(
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        display_mode="hdmi"
    )

    pl.create(sensor=Sensor(id=0))

    face_det = FaceDetectionApp(
        kmodel_path,
        [320,320],
        anchors,
        rgb888p_size=rgb888p_size,
        display_size=display_size
    )
    face_det.config_preprocess()

    # 初始化舵机
    pwm_x, pwm_y = servo_init()

    # 回中
    set_servo_angle(pwm_x, 90)
    set_servo_angle(pwm_y, 90)
    utime.sleep_ms(800)

    try:
        while True:
            os.exitpoint()

            img = pl.get_frame()
            dets = face_det.run(img)

            face_det.draw_result(pl, dets)

            face = face_det.get_largest_face(dets)
            if face:
                x,y,w,h = face[:4]
                cx = x + w//2
                cy = y + h//2

                # ===== 绝对映射（核心思想）=====
                pan  = 180 - cx / rgb888p_size[0] * 180

                tilt = 180 - cy / rgb888p_size[1] * 180

                pan  = clamp(pan,  PAN_MIN,  PAN_MAX)
                tilt = clamp(tilt, TILT_MIN, TILT_MAX)

                set_servo_angle(pwm_x, pan)
                set_servo_angle(pwm_y, tilt)

            pl.show_image()
            gc.collect()
            utime.sleep_ms(20)

    except Exception as e:
        print("运行异常:", e)

    finally:
        pwm_x.deinit()
        pwm_y.deinit()
        face_det.deinit()
        pl.destroy()
