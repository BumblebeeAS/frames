
#include "bb_planner_msgs/srv/get_pose_to_dest_frame.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::placeholders;
using namespace std::chrono;
using namespace std::chrono_literals;

class DestPoseFromTf : public rclcpp::Node
{
public:
	DestPoseFromTf() :
		Node("dest_pose_from_tf")
	{
		RCLCPP_INFO(this->get_logger(), "Node initiated....");
		tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
		srv_ = this->create_service<bb_planner_msgs::srv::GetPoseToDestFrame>(
			"get_dest_pose", 
			std::bind(&DestPoseFromTf::srvCallBack, this, _1, _2));
	}

private:
	rclcpp::Service<bb_planner_msgs::srv::GetPoseToDestFrame>::SharedPtr srv_;
	std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
	void srvCallBack(
		const std::shared_ptr<bb_planner_msgs::srv::GetPoseToDestFrame::Request> request,
		const std::shared_ptr<bb_planner_msgs::srv::GetPoseToDestFrame_Response> response
	);
};

void DestPoseFromTf::srvCallBack(
	const std::shared_ptr<bb_planner_msgs::srv::GetPoseToDestFrame::Request> request, 
	const std::shared_ptr<bb_planner_msgs::srv::GetPoseToDestFrame_Response> response
)
{
	std::string dest_frame_name = request->dest_frame_name;
	geometry_msgs::msg::PoseStamped input_pose = request->input_pose;
	geometry_msgs::msg::PoseStamped dest_pose;
	try
	{
		tf_buffer_->transform(input_pose, dest_pose, dest_frame_name);
		response->output_pose = dest_pose;
		response->tf_success = true;
		tf_buffer_->clear();
	}
	catch (tf2::TransformException& ex) {
		RCLCPP_ERROR(this->get_logger(), "Failure %s\n", ex.what());
		RCLCPP_ERROR(this->get_logger(), "Retaining input pose");
		response->output_pose = input_pose;
		response->tf_success = false;
	}
}

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<DestPoseFromTf>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}