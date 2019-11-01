from random import shuffle
import glob
import sys
import cv2
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addr):
    img = cv2.imread(addr)
    if img is None:
        return None
    # Resize the image but cv2 load images as BGR
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    # Convert images to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def createDataRecord(out_path_TFRecord, addrs, labels):
    # Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_path_TFRecord)
    for iterat in range(len(addrs)):
        if iterat % 1000 == 0:
            print("Train data: {}/{}".format(iterat, len(addrs)))
            sys.stdout.flush()
        img = load_image(addrs[iterat])
        label = labels[iterat]
        if img is None:
            continue
        # Create a feature, a dict with the data to store in the TFRecords file
        dict_feature_data = {
            "image_raw": _bytes_feature(img.tostring()),
            "label": _int64_feature(label)
        }
        # Wrap the data as TensorFlow Features.
        feature = tf.train.Features(feature=dict_feature_data)
        # Create an example protocol buffer
        example = tf.train.Example(features=feature)
        # Serialize the data to string
        serialized = example.SerializeToString()
        # Write the serialized data to the TFRecords file
        writer.write(serialized)
    writer.close()
    sys.stdout.flush()


# Put all the path of the .jpg and the .JPEG files inside a list
furniture_train_path_JPEG = "furniture/*/*.JPEG"
furniture_train_path_jpg = "furniture/*/*.jpg"
addrs_JPEG = glob.glob(furniture_train_path_JPEG)
addrs_jpg = glob.glob(furniture_train_path_jpg)

# Concatenate those lists
addrs = addrs_JPEG + addrs_jpg

# Create a list of labels with int
labels = []
for addr in addrs:
    if "Bathtub" in addr:
        labels.append(0)
    elif "Chair" in addr:
        labels.append(1)
    elif "CoffeMaker" in addr:
        labels.append(2)
    elif "Desk" in addr:
        labels.append(3)
    elif "Dishwasher" in addr:
        labels.append(4)
    elif "Door" in addr:
        labels.append(5)
    elif "Mouse" in addr:
        labels.append(6)
    else:
        labels.append(7)

# Shuffle data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

# Create 3 TFRecords : path to the image + label of this image
createDataRecord('train.tfrecords', train_addrs, train_labels)
createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('test.tfrecords', test_addrs, test_labels)
