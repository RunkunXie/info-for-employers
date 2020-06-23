import os
import numpy as np
import h5py
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)

class DataManager(object):
    """
    Class for data preprocessing
    """
    def __init__(self, main_path, dataset_type='train'):
        self.dataset_type = dataset_type
        self.path_to_imgs = [os.path.join(main_path, dataset_type, 'image', d) for d in os.listdir(os.path.join(main_path, dataset_type, 'image'))]
        self.path_to_struct_mat = os.path.join(main_path, dataset_type, 'mat', 'digitStruct.mat')
        self.n_samps = len(self.path_to_imgs)
        self.pointer = 0
    
    @staticmethod
    def get_attrs(struct_mat, index):
        """
        Return a dictionary with keys: label, left, top, width, and height
        Each key is paired with multiple values
        """
        attrs = {}
        item = struct_mat['digitStruct']['bbox'][index].item()
        keys = ['label', 'left', 'top', 'width', 'height']
        assert all([k in struct_mat[item].keys() for k in keys])
        for k in keys:
            attr = struct_mat[item][k]
            values = [struct_mat[attr.value[i].item()].value[0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            attrs[k] = values
        return attrs
    
    @staticmethod
    def expand_and_crop(image, bbox_left, bbox_top, bbox_width, bbox_height):
        """
        Expand and crop the dataset, as is suggested in the paper
        """
        # the following codes of this function are referenced from online resources
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image
    
    @staticmethod
    def correct_labels(labels):
        """
        Consolidate the form of the digits
        Digit 10 represents no digit
        """
        digits = [10, 10, 10, 10, 10]
        for idx, label in enumerate(labels):
            digits[idx] = int(label if label != 10 else 0)
        return digits
        
    def read_and_proc(self, struct_mat):
        """
        Read and convert to example
        Return None if no data is available
        """
        if self.pointer == self.n_samps: return None
        path_to_img = self.path_to_imgs[self.pointer]
        index = int(path_to_img.split('/')[-1].split('.')[0]) - 1
        self.pointer += 1
        attrs = DataManager.get_attrs(struct_mat, index)
        labels = attrs['label']
        n_digits = len(labels)
        if n_digits > 5:
            return self.read_and_proc(struct_mat)
        digits = DataManager.correct_labels(labels)
        # the following codes of this function are referenced from online resources
        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(DataManager.expand_and_crop(Image.open(path_to_img), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_digits])),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
        }))
        return example
    
    def write_tfrecord(self, path_to_tfrecord, force=False):
        """
        Convert raw data to tfrecord format, to be used later in training
        """
        # Note that the idea to write data into tfrecord is learned from online resources, but codes are composed by ourselves
        import sys
        recur_lmt = sys.getrecursionlimit()
        sys.setrecursionlimit(3500)
        if tf.gfile.Exists(path_to_tfrecord) and not force:
            logger.info('--- tfrecord for {} dataset already exists, no writting process will be created ---'.format(path_to_tfrecord.split('/')[-1].split('.')[0]))
            return
        try:
            logger.info('--- {} images found to be written into tfrecord ---'.format(self.n_samps))
            writer = tf.python_io.TFRecordWriter(path_to_tfrecord)
            with h5py.File(self.path_to_struct_mat, 'r') as struct_mat:
                for i in tqdm(range(self.n_samps)):
                    example = self.read_and_proc(struct_mat)
                    if example is None: break
                    writer.write(example.SerializeToString())
            writer.close()
            logger.info('--- tfrecord for {} dataset are successfully completed ---'.format(path_to_tfrecord.split('/')[-1].split('.')[0]))
        except RuntimeError as ex:
            sys.setrecursionlimit(recur_lmt)
            raise ex
        return
    
    @staticmethod
    def augmentation(image):
        """
        Data augmentation
        """
        logger.info('--- augmenting images ---')
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [54, 54, 3])
        return image
    
    @staticmethod
    def read_and_decode(tfrecord_queue):
        """
        Read data from the tfrecord file
        """
        logger.info('--- read and decode tfrecord ---')
        reader = tf.TFRecordReader()
        _, tfrecord_serialized = reader.read(tfrecord_queue)
        features = tf.parse_single_example(tfrecord_serialized,
                                           features={'image': tf.FixedLenFeature([], tf.string),
                                                     'length': tf.FixedLenFeature([], tf.int64),
                                                     'digits': tf.FixedLenFeature([5], tf.int64)},
                                           name='features')
        
        image = DataManager.augmentation(tf.decode_raw(features['image'], tf.uint8))
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)
        return image, length, digits
    
    def data_generator(self, path_to_tfrecord, batch_size, is_shuffle=False):
        """
        Generate data, containing image, length and digits
        """
        assert tf.gfile.Exists(path_to_tfrecord), '{} not found'.format(path_to_tfrecord)
        logger.info('--- generating {} samples from {} dataset ---'.format(batch_size, self.dataset_type))
        tfrecord_queue = tf.train.string_input_producer([path_to_tfrecord], name='queue')
        image, length, digits = DataManager.read_and_decode(tfrecord_queue)
        min_queue_samps = int(0.3 * self.n_samps)
        if is_shuffle:
            image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length, digits],
                                                                             batch_size=batch_size,
                                                                             num_threads=2,
                                                                             capacity=min_queue_samps + 3 * batch_size,
                                                                             min_after_dequeue=min_queue_samps)
        else:
            image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_samps + 3 * batch_size)
        return image_batch, length_batch, digits_batch
    