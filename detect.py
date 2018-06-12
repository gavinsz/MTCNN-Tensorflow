# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import time

import tensorflow as tf
import cv2
import numpy as np
from cv2_detector import Cv2FaceDetector
from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames, extract_image_chips, align

class FaceDectect:
    def __init__(self):
        """
            Initialize the detector

            Parameters:
            ----------
                minsize : float number
                    minimal face to detect
                threshold : float number
                    detect threshold for 3 stages
                factor: float number
                    scale factor for image pyramid
        """
        self.minsize = 10
        self.factor = 0.7
        self.threshold = [0.6, 0.7, 0.8]
        self.model_dir = 'save_model/all_in_one'
        self.sess = tf.Session()
        self.cv2_detector = Cv2FaceDetector()

        file_paths = get_model_filenames(self.model_dir)
        if len(file_paths) == 3:
            image_pnet = tf.placeholder(
                tf.float32, [None, None, None, 3])
            pnet = PNet({'data': image_pnet}, mode='test')
            out_tensor_pnet = pnet.get_all_output()

            image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
            rnet = RNet({'data': image_rnet}, mode='test')
            out_tensor_rnet = rnet.get_all_output()

            image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
            onet = ONet({'data': image_onet}, mode='test')
            out_tensor_onet = onet.get_all_output()

            saver_pnet = tf.train.Saver(
                            [v for v in tf.global_variables()
                            if v.name[0:5] == "pnet/"])
            saver_rnet = tf.train.Saver(
                            [v for v in tf.global_variables()
                            if v.name[0:5] == "rnet/"])
            saver_onet = tf.train.Saver(
                            [v for v in tf.global_variables()
                            if v.name[0:5] == "onet/"])

            saver_pnet.restore(self.sess, file_paths[0])

            def pnet_fun(img): return self.sess.run(
                out_tensor_pnet, feed_dict={image_pnet: img})

            saver_rnet.restore(self.sess, file_paths[1])

            def rnet_fun(img): return self.sess.run(
                out_tensor_rnet, feed_dict={image_rnet: img})

            saver_onet.restore(self.sess, file_paths[2])

            def onet_fun(img): return self.sess.run(
                out_tensor_onet, feed_dict={image_onet: img})

        else:
            saver = tf.train.import_meta_graph(file_paths[0])
            saver.restore(self.sess, file_paths[1])
    
    def __del__(self):
        self.sess.close()

    def pnet_fun(self, img):
        return self.sess.run(
                            ('softmax/Reshape_1:0',
                            'pnet/conv4-2/BiasAdd:0'),
                            feed_dict={'Placeholder:0': img})

    def rnet_fun(self, img):
        return self.sess.run(
                       ('softmax_1/softmax:0',
                        'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={'Placeholder_1:0': img})

    def onet_fun(self, img):
        return self.sess.run(
                            ('softmax_2/softmax:0',
                            'onet/conv6-2/onet/conv6-2:0',
                            'onet/conv6-3/onet/conv6-3:0'),
                            feed_dict={'Placeholder_2:0': img})

    def display(self, img, rectangles, points, save_name=None):
        for rectangle in rectangles:
            cv2.putText(img, str(rectangle[4]),
                        (int(rectangle[0]), int(rectangle[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0))
            cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])),
                            (int(rectangle[2]), int(rectangle[3])),
                            (255, 0, 0), 1)
        for point in points:    
            for i in range(0, 10, 2):
                cv2.circle(img, (int(point[i]), int(
                    point[i + 1])), 2, (0, 255, 0))

        cv2.imshow("test", img)
        if save_name is not None:
            cv2.imwrite(save_name, img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def gen_5pt(self, img, save_name):
        rectangles, points = detect_face(img, self.minsize,
                            self.pnet_fun, self.rnet_fun, self.onet_fun,
                            self.threshold, self.factor)
        if rectangles.shape[0] > 0:
            points = np.transpose(points)
            #print landmake
            for p in points:
                #x0, y0, x1, y1....
                five_pt_array = np.array([(p[0], p[1]), (p[2], p[3]), (p[4], p[5]), (p[6], p[7]), (p[8], p[9])])
                np.savetxt(save_name, five_pt_array, fmt='%d', newline='\n')

    def save_5pt(self, p, save_name):
        #x0, y0, x1, y1....
        five_pt_array = np.array([(p[0], p[1]), (p[2], p[3]), (p[4], p[5]), (p[6], p[7]), (p[8], p[9])])
        np.savetxt(save_name, five_pt_array, fmt='%d', newline='\n')

    def detect(self, full_path, file_name):
        img = cv2.imread(full_path)
        succ, _ = self.cv2_detector.face_detected(img)
        start_time = time.time()
        rectangles, points = detect_face(img, self.minsize,
                                         self.pnet_fun, self.rnet_fun, self.onet_fun,
                                         self.threshold, self.factor)
        points = np.transpose(points)                                         
        duration = time.time() - start_time
        
        print('face detect cost=%ds|totol faces=%d' %(duration, rectangles.shape[0]))
        #print('points=', points)

        if rectangles.shape[0] > 0:
            i = 0
            for point in points:
                chip, key_points = align(img, point, 128, 48, 40)
                key_points = np.resize(key_points, 10)
                print('key_points=', key_points)
                self.save_5pt(key_points, 'tmp/'+str(i+603)+file_name+'.5pt')
                cv2.imwrite('tmp/'+str(i+603)+file_name+'.png', chip)
                i += 1
            '''
            chips = extract_image_chips(img, points, 128, 0.39)
            for i, chip in enumerate(chips):
                succ, _ = self.cv2_detector.face_detected(chip)
                if True is True:
                    #cv2.imshow('chip_'+str(i), chip)
                    cv2.imwrite('tmp/'+str(i+603)+file_name+'.png', chip)
                    self.gen_5pt(chip, 'tmp/'+str(i+603)+file_name+'.5pt')
                else:
                    print('chip_'+ str(i) + ' not deteced face')
            '''

        #self.display(img,rectangles, points)
        return rectangles, points

face_detector = FaceDectect()

if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    #face_detector = FaceDectect()
    face_detector.detect('images/613_04_01_190_12_cropped.jpg', '_613_04_01_190_12_cropped')
    #face_detector.detect('images/611_04_01_190_12.jpg', '613_04_01_190_12_cropped')
    #face_detector.detect('images/689_01_01_041_12.png', '689_01_01_041_12')
    #face_detector.detect('images/03.jpg', '03_')
    #face_detector.detect('images/38480_1528181355018882.jpg', '_04_01_190_12_cropped