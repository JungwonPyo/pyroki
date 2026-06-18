#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from scene_understanding_msgs.msg import (
    SceneContext,
    DetectedObject3D,
    BoundingBox3D,
    CameraModel,
    SituationHypothesis,
)


class FakeScenePublisher(Node):
    def __init__(self):
        super().__init__("fake_scene_publisher")
        self.pub = self.create_publisher(SceneContext, "/scene_context", 10)
        self.timer = self.create_timer(0.1, self.publish_scene)
        self.t = 0.0
        self.get_logger().info("Publishing fake perceived objects on /scene_context")

    def make_object(self, object_id, class_name, center, size, frame_id="camera_link", score=0.99):
        obj = DetectedObject3D()
        obj.id = object_id
        obj.class_name = class_name
        obj.score = score
        obj.bbox_2d_xyxy = [100, 100, 200, 200]

        bbox = BoundingBox3D()
        bbox.valid = True
        bbox.frame_id = frame_id
        bbox.center.x, bbox.center.y, bbox.center.z = map(float, center)
        bbox.size.x, bbox.size.y, bbox.size.z = map(float, size)

        bbox.min_corner.x = bbox.center.x - bbox.size.x / 2.0
        bbox.min_corner.y = bbox.center.y - bbox.size.y / 2.0
        bbox.min_corner.z = bbox.center.z - bbox.size.z / 2.0
        bbox.max_corner.x = bbox.center.x + bbox.size.x / 2.0
        bbox.max_corner.y = bbox.center.y + bbox.size.y / 2.0
        bbox.max_corner.z = bbox.center.z + bbox.size.z / 2.0
        bbox.z_median = bbox.center.z
        bbox.z_mean = bbox.center.z
        bbox.z_min = bbox.min_corner.z
        bbox.z_max = bbox.max_corner.z
        bbox.z_std = 0.01
        bbox.method = "fake_test_box"

        obj.bbox_3d = bbox
        return obj

    def publish_scene(self):
        self.t += 0.1

        msg = SceneContext()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.scene_id = "test_scene"
        msg.planner_frame = "base_link"

        msg.camera = CameraModel()
        msg.camera.width = 640
        msg.camera.height = 480
        msg.camera.k = [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]
        msg.camera.distortion_model = "plumb_bob"
        msg.camera.d = []

        msg.situation = SituationHypothesis()
        msg.situation.label = "recognized_objects_only"
        msg.situation.index = 0
        msg.situation.confidence = 0.95
        msg.situation.labels = ["recognized_objects_only"]
        msg.situation.probs = [0.95]
        
        # Sample target object in camera frame
        target_center = [0.3, 0.45, 0.15]
        target_size = [0.04, 0.04, 0.08]

        # Moving obstacle 1
        obs1_center = [-0.05, 0.42 + 0.05 * math.sin(self.t), 0.15]
        obs1_size = [0.08, 0.08, 0.30]

        # Static obstacle 2
        obs2_center = [0.28, 0.22, 0.10]
        obs2_size = [0.10, 0.12, 0.20]

        msg.objects = [
            self.make_object("target_box_01", "target_part", target_center, target_size),
            self.make_object("obstacle_person_01", "person", obs1_center, obs1_size),
            self.make_object("obstacle_box_01", "obstacle_box", obs2_center, obs2_size),
        ]
        msg.relationships = []

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeScenePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()