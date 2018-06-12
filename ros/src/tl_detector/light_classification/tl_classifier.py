from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import rospy

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        graph_path = './graph/frozen_inference_graph.pb'
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # expect rgb.
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                rospy.loginfo('image.shape: {}'.format(image.shape))
                # Convert image format.
                # image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                rospy.loginfo('image_np_expanded.shape: {}'.format(image_np_expanded.shape))

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                rospy.loginfo(
                    'num boxes: {} num scores: {} num classes: {}'.
                    format(len(boxes),len(scores), len(classes)))
                green_score = 0 # id=1
                red_score = 0 # id=2
                yellow_score = 0 # id =3

                # FIXME: Imporve logic.
                for i, cl in enumerate(classes):
                    score = scores[i]
                    if cl is 1:
                        green_score = green_score + score
                    if cl is 2:
                        red_score = red_score + score
                    if cl is 3:
                        yellow_score = yellow_score + score
                scores = np.array([green_score, red_score, yellow_score])
                idx = np.argsort(scores)[-1]
                rospy.loginfo('detected light idx: {}'.format(idx))

                if idx == 0:
                    return TrafficLight.GREEN
                if idx == 1:
                    return TrafficLight.RED
                if idx == 2:
                    return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
