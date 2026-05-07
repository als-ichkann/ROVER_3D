from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    use_sim_time_raw = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time = str(use_sim_time_raw).lower() in ("true", "1", "yes", "on")

    # 1) Map fusion node
    map_fusion_node = Node(
        package="map_fusion",
        executable="global_map_publisher.py",
        name="global_map_node",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "bot_prefix": "bot",                    # matches bot1/..., bot2/...
            "bot_cloud_topic": "/cloud_registered",  # /<bot_prefix><i>/<bot_cloud_topic> PointCloud2
            "bot_cloud_frame": "world",             # <bot_prefix><i>/<bot_cloud_frame>

            "publish_rate_hz": 0.5,
            "voxel_leaf_size": 0.1,
            "filter_high_intensity": True,
            "intensity_max_threshold": 1000.0,
            "filter_body_points": True,
            "body_frame": "base_link",
            "body_filter_radius": 0.45,

            "combined_map_topic": "/global_downsampled_map",
            "origin_frame": "map_origin",
        }],
    )

    # 2) Odom publisher
    global_odom_node = Node(
        package="map_fusion",
        executable="global_odom_publisher.py",
        name="global_odom_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "global_frame": "map_origin",   # map_origin -> bot<i>/<bot_frame>
            "bot_prefix": "bot",            # matches bot1/..., bot2/...
            "bot_frame": "base_link",       # <bot_prefix><i>/<bot_frame>
            "publish_rate_hz": 10.0,
            "odom_topic_suffix": "global_odom",
            "source_odom_topic": "lidar_slam/odom",
        }],
    )

    return [map_fusion_node, global_odom_node]


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation time"),
        OpaqueFunction(function=_launch_setup),
    ])
