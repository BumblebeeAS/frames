"""Do not report successful or partial conversions after transform failures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bb_planner_msgs.srv import GetPoseToControlsFrame
from geometry_msgs.msg import PoseStamped

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
