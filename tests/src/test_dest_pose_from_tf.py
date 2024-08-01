import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from bb_planner_msgs.srv import GetPoseToDestFrame
from tf2_ros import TransformBroadcaster
import tf_transformations


class TestDestPoseFromTf(Node):

    def __init__(self):
        super().__init__('test_dest_pose_from_tf')

        # Create a TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create a timer to publish fake transforms
        self.timer = self.create_timer(0.1, self.publish_fake_transform)
        self.i = 0

        # Wait for the service to be available
        self.client = self.create_client(GetPoseToDestFrame, '/asv4/get_dest_pose')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

        # Call the service after some delay to allow transforms to be published
        self.create_timer(2.0, self.call_service)

    def publish_fake_transform(self):
        if self.i > 4:
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'source_frame'
            transform.child_frame_id = 'dest_frame'
            transform.transform.translation.x = 1.0
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 0.0
            quat = tf_transformations.quaternion_from_euler(0, 0, 0)
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(transform)

    def call_service(self):
        request = GetPoseToDestFrame.Request()
        request.dest_frame_name = 'dest_frame'

        input_pose = PoseStamped()
        input_pose.header.stamp = (
            self.get_clock().now() - Duration(seconds=2.0)).to_msg()
        input_pose.header.frame_id = 'source_frame'
        input_pose.pose.position.x = 0.0
        input_pose.pose.position.y = 0.0
        input_pose.pose.position.z = 0.0
        input_pose.pose.orientation.w = 1.0
        input_pose.pose.orientation.x = 0.0
        input_pose.pose.orientation.y = 0.0
        input_pose.pose.orientation.z = 0.0

        request.input_pose = input_pose
        request.timeout = 0.5

        future = self.client.call_async(request)
        future.add_done_callback(self.service_callback)
        self.i += 1

    def service_callback(self, future):
        try:
            response = future.result()
            if response.tf_success:
                self.get_logger().info(f'Transformation successful: {response.output_pose}')
            else:
                self.get_logger().error('Transformation failed')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TestDestPoseFromTf()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
