import os
import io
from lxml import etree
from PIL import Image
import tensorflow.compat.v1 as tf
from models.research.object_detection.utils import dataset_util
from models.research.object_detection.utils import label_map_util

# Rest of your script remains unchanged
def create_tf_example(image_path, annotation_path, label_map_dict):
    with tf.compat.v1.gfile.GFile(annotation_path, 'r') as fid:
        xml_str = fid.read()

    # Parse XML annotation
    xml = etree.fromstring(xml_str)

    # Read the image
    with tf.compat.v1.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # Parse bounding box coordinates
    xmin = int(xml.find('.//xmin').text)
    ymin = int(xml.find('.//ymin').text)
    xmax = int(xml.find('.//xmax').text)
    ymax = int(xml.find('.//ymax').text)

    # Get image size
    image = Image.open(image_path)
    width, height = image.size

    # Get label id using label_map_dict
    label = xml.find('.//name').text
    label_id = label_map_dict[label]

    # Create TF example
    tf_example = tf.compat.v1.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature([xmin / width]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([xmax / width]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([ymin / height]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([ymax / height]),
        'image/object/class/text': dataset_util.bytes_feature(label.encode('utf8')),
        'image/object/class/label': dataset_util.int64_list_feature([label_id]),
    }))

    return tf_example

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.compat.v1.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords

def main(data_dir, output_dir, label_map_path, contextlib2=None):
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    annotation_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')

    annotation_files = os.listdir(annotation_dir)

    num_shards = 10
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, os.path.join(output_dir, 'output'), num_shards)

        for idx, annotation_file in enumerate(annotation_files):
            annotation_path = os.path.join(annotation_dir, annotation_file)
            image_filename = os.path.splitext(annotation_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_filename)

            tf_example = create_tf_example(image_path, annotation_path, label_map_dict)
            output_tfrecords[idx % num_shards].write(tf_example.SerializeToString())

if __name__ == '__main__':
    main(
        data_dir='C:/Users/HP/Documents/Capstone/dataset',
        output_dir='C:/Users/HP/Documents/Capstone/output',
        label_map_path='C:/Users/HP/Documents/Capstone/dataset/label_map.pbtxt'
    )
