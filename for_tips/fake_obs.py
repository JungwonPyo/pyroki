#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from scene_understanding_msgs.msg import (
    SceneContext,
    DetectedObject3D,
    BoundingBox3D,
    CameraModel,
    SituationHypothesis,
    SceneRelation,
)


class FakeScenePublisher(Node):
    def __init__(self):
        super().__init__("fake_scene_publisher")
        self.pub = self.create_publisher(SceneContext, "/scene_context", 10)
        self.timer = self.create_timer(0.1, self.publish_scene)

        self.t = 0.0
        self.get_logger().info("Publishing fake SceneContext on /scene_context")

    def publish_scene(self):
        msg = SceneContext()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_link0"

        msg.scene_id = "test_scene"
        # msg.planner_frame = "panda_link0"
        msg.planner_frame = "base_link"

        msg.camera = CameraModel()
        msg.camera.width = 640
        msg.camera.height = 480
        msg.camera.k = [525.0, 0.0, 320.0,
                        0.0, 525.0, 240.0,
                        0.0, 0.0, 1.0]
        msg.camera.distortion_model = "plumb_bob"
        msg.camera.d = []

        msg.situation = SituationHypothesis()
        msg.situation.label = "obstacle_on_path"
        msg.situation.index = 0
        msg.situation.confidence = 0.95
        msg.situation.labels = ["obstacle_on_path"]
        msg.situation.probs = [0.95]

        obj = DetectedObject3D()
        obj.id = "fake_person_01"
        obj.class_name = "person"
        obj.score = 0.99
        obj.bbox_2d_xyxy = [200, 120, 300, 360]

        bbox = BoundingBox3D()
        bbox.valid = True
        # bbox.frame_id = "panda_link0"
        bbox.frame_id = "base_link"

        # Static obstacle near your waypoint corridor
        bbox.center.x = -0.45
        bbox.center.y = 0.02
        bbox.center.z = 0.25

        bbox.size.x = 0.01
        bbox.size.y = 0.01
        bbox.size.z = 0.01

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
        msg.objects = [obj]
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