#ifndef ENU_NED_CONVERTER_HPP
#define ENU_NED_CONVERTER_HPP

#include <functional>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include "enu_ned_converter_helper.hpp"  // Assuming this header defines transform_msg

template<typename MessageType>
class EnuNedRepublisher : public rclcpp::Node
{
public:
    EnuNedRepublisher(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
    : Node("enu_ned_republisher", options)
    {
        this->declare_parameter<std::string>("topic", "input_topic");
        this->declare_parameter<std::string>("output_topic", "output_topic");
        this->declare_parameter<std::string>("source_frame", "source_frame");
        this->declare_parameter<bool>("is_body", false);
        this->declare_parameter<bool>("transform_child", false);

        this->get_parameter("topic", topic_);
        this->get_parameter("output_topic", output_topic_);
        this->get_parameter("source_frame", source_frame_);
        this->get_parameter("is_body", is_body_);
        this->get_parameter("transform_child", transform_child_);

        publisher_ = this->create_publisher<MessageType>(output_topic_, 10);
        subscription_ = this->create_subscription<MessageType>(
            topic_, 10, std::bind(&EnuNedRepublisher::listener_callback, this, std::placeholders::_1));
    }

private:
    std::string topic_;
    std::string output_topic_;
    std::string source_frame_;
    bool is_body_;
    bool transform_child_;

    typename rclcpp::Publisher<MessageType>::SharedPtr publisher_;
    typename rclcpp::Subscription<MessageType>::SharedPtr subscription_;

    void listener_callback(const typename MessageType::SharedPtr msg)
    {
        if (check_msg(msg))
        {
            auto transformed_msg = handle(*msg, is_body_, transform_child_);
            if (transformed_msg)
            {
                publisher_->publish(*transformed_msg);
            }
        }
    }

    bool check_msg(const typename MessageType::SharedPtr& msg_ptr)
    {
        const auto& msg = *msg_ptr;
        if (msg.header.frame_id != source_frame_)
        {
            RCLCPP_WARN(this->get_logger(), "Received message with frame_id %s but expected %s.", msg.header.frame_id.c_str(), source_frame_.c_str());
            return false;
        }
        return true;
    }
    std::shared_ptr<MessageType> handle(const MessageType& msg, bool is_body, bool transform_child)
    {
        // Create a copy of the message
        std::shared_ptr<MessageType> msg_copy = std::make_shared<MessageType>(msg);
        transform_msg(*msg_copy, is_body_, transform_child_);
        return msg_copy;
    }
};

#endif // ENU_NED_CONVERTER_HPP
