"""Do not report successful or partial conversions after transform failures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bb_planner_msgs.srv import GetPoseToControlsFrame
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import pytest

spec = importlib.util.spec_from_file_location(
    'controls_converter', Path(__file__).parents[1] / 'scripts/convert_to_controls_pose.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_transform_fails_without_output():
    node = SimpleNamespace(
        controls_frame='home_ned', base_frame='uav/base_link_frd',
        get_logger=Mock(), transform_to_frame=Mock(return_value=None),
    )
    request = GetPoseToControlsFrame.Request()
    request.input_poses = [PoseStamped()]
    response = module.ConvertToControlsPose.convert_callback(
        node, request, GetPoseToControlsFrame.Response()
    )
    assert not response.tf_success
    assert not response.output_poses


def test_vehicle_transform_uses_odometry_in_both_directions():
    odom = Odometry()
    odom.header.frame_id = 'home_ned'
    odom.header.stamp.sec = 123
    odom.child_frame_id = 'uav/base_link_frd'
    odom.pose.pose.position.x = 4.0
    odom.pose.pose.position.y = 2.0
    odom.pose.pose.orientation.w = 1.0
    node = SimpleNamespace(prefer_odometry=True, odom_buffer=[odom], tf_buffer=Mock())
    forward = module.ConvertToControlsPose.lookup_with_fallback(
        node, 'home_ned', 'uav/base_link_frd', 1.0)
    inverse = module.ConvertToControlsPose.lookup_with_fallback(
        node, 'uav/base_link_frd', 'home_ned', 1.0)
    assert forward.transform.translation.x == pytest.approx(4.0)
    assert inverse.transform.translation.x == pytest.approx(-4.0)
    assert forward.header == odom.header
    assert inverse.child_frame_id == 'home_ned'
    assert inverse.header.frame_id == 'uav/base_link_frd'
    assert odom.header.frame_id == 'home_ned'
    node.tf_buffer.lookup_transform.assert_not_called()


def test_tf_remains_preferred_when_odometry_is_disabled():
    odom = Odometry()
    odom.header.frame_id = 'odom_ned'
    odom.child_frame_id = 'auv5/base_link_frd'
    node = SimpleNamespace(
        prefer_odometry=False, odom_buffer=[odom], tf_buffer=Mock(),
        get_clock=Mock(),
    )
    result = module.ConvertToControlsPose.lookup_with_fallback(
        node, 'odom_ned', 'auv5/base_link_frd', 1.0)
    assert result is node.tf_buffer.lookup_transform.return_value
    node.tf_buffer.lookup_transform.assert_called_once()
