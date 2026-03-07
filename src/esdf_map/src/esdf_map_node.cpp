#include "esdf_map/esdf_map_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <utility>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/create_timer.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2/exceptions.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

namespace esdf_map
{

    EsdfMapNode::EsdfMapNode(const rclcpp::NodeOptions &options)
        : rclcpp::Node("esdf_map_node", options)
    {
        declareAndLoadParams();

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        EsdfMapCore::Config cfg = makeCoreConfig();
        core_ = std::make_unique<EsdfMapCore>(cfg);

        bots_.clear();
        if (input_mode_ == "robots") {
            buildBotList();
        }

        esdf_grid_pub_ = create_publisher<PointCloud2>(
            "/esdf/grid", rclcpp::QoS(1).transient_local());
        esdf_grid_roi_pub_ = create_publisher<PointCloud2>(
            "/esdf/grid_roi", rclcpp::QoS(1).transient_local());
        costmap_pub_ = create_publisher<OccupancyGrid>(
            "/esdf/costmap_2d", rclcpp::QoS(1).transient_local());

        setupSubscriptions();
        setupTimers();
        setupService();

        if (input_mode_ == "robots") {
            RCLCPP_INFO(get_logger(), "esdf_map_node initialized in ROBOTS mode with %zu bots", bots_.size());
        } else {
            RCLCPP_INFO(get_logger(), "esdf_map_node initialized in CLOUD mode: topic=%s frame=%s",
                        cloud_topic_.c_str(), cloud_frame_.c_str());
        }
    }

    void EsdfMapNode::declareAndLoadParams() {
        input_mode_   = declare_parameter<std::string>("input_mode", "robots");
    
        cloud_topic_  = declare_parameter<std::string>("cloud_topic", "/global_downsampled_map");
        cloud_frame_  = declare_parameter<std::string>("cloud_frame", "map_origin");


        bot_prefix_ = declare_parameter<std::string>("bot_prefix", "bot");
        bot_cloud_topic_ = declare_parameter<std::string>("bot_cloud_topic", "cloud_registered");
        bot_cloud_frame_ = declare_parameter<std::string>("bot_cloud_frame", "world");
        bot_sensor_frame_ = declare_parameter<std::string>("bot_sensor_frame", "lidar_link");

        world_frame_ = declare_parameter<std::string>("world_frame", "map_origin");
        tf_timeout_sec_ = declare_parameter<double>("tf_timeout_sec", 0.1);

        bot_ids_ = declare_parameter<std::vector<int64_t>>("bot_ids", {1});

        esdf_resolution_ = declare_parameter<double>("esdf_resolution", 0.1);
        map_size_x_ = declare_parameter<double>("map_size_x", 30.0);
        map_size_y_ = declare_parameter<double>("map_size_y", 30.0);
        map_size_z_ = declare_parameter<double>("map_size_z", 5.0);
        map_origin_x_ = declare_parameter<double>("map_origin_x", -15.0);
        map_origin_y_ = declare_parameter<double>("map_origin_y", -15.0);
        map_origin_z_ = declare_parameter<double>("map_origin_z", 0.0);

        max_ray_length_ = declare_parameter<double>("max_ray_length", 30.0);
        truncation_distance_ = declare_parameter<double>("truncation_distance", 0.3);
        esdf_max_distance_ = declare_parameter<double>("esdf_max_distance", 5.0);
        enable_chamfer_relax_ = declare_parameter<bool>("enable_chamfer_relax", true);

        integrate_every_cloud_ = declare_parameter<bool>("integrate_every_cloud", true);
        esdf_update_rate_hz_ = declare_parameter<double>("esdf_update_rate_hz", 2.0);

        publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 1.0);
        publish_full_grid_ = declare_parameter<bool>("publish_full_grid", true);
        publish_roi_grid_ = declare_parameter<bool>("publish_roi_grid", true);
        publish_costmap_2d_ = declare_parameter<bool>("publish_costmap_2d", true);
        grid_max_distance_ = declare_parameter<double>("grid_max_distance", 2.0);
        costmap_layer_z_ = declare_parameter<double>("costmap_layer_z", 0.5);
        costmap_free_distance_ = declare_parameter<double>("costmap_free_distance", 0.5);
        costmap_lethal_distance_ = declare_parameter<double>("costmap_lethal_distance", 0.1);
        time_log_ = declare_parameter<bool>("time_log", false);
        debug_profile_ = declare_parameter<bool>("debug_profile", false);
        debug_profile_every_ = declare_parameter<int>("debug_profile_every", 20);
    }

    EsdfMapCore::Config EsdfMapNode::makeCoreConfig() const {
        EsdfMapCore::Config cfg;
        cfg.resolution = esdf_resolution_;
        cfg.origin = Vec3d(map_origin_x_, map_origin_y_, map_origin_z_);

        auto nx = static_cast<int>(std::round(map_size_x_ / esdf_resolution_));
        auto ny = static_cast<int>(std::round(map_size_y_ / esdf_resolution_));
        auto nz = static_cast<int>(std::round(map_size_z_ / esdf_resolution_));
        cfg.dims = Vec3i(std::max(nx, 1), std::max(ny, 1), std::max(nz, 1));

        cfg.max_ray_length = max_ray_length_;
        cfg.truncation_distance = truncation_distance_;
        cfg.esdf_max_distance = esdf_max_distance_;
        cfg.enable_chamfer_relax = enable_chamfer_relax_;
        return cfg;
    }

    void EsdfMapNode::buildBotList() {
        bots_.clear();
        if (bot_ids_.empty()) {
            RCLCPP_WARN(get_logger(), "No bot_ids specified, nothing will be handled.");
            return;
        }

        for (int id : bot_ids_) {
            Bot bot;
            bot.id = id;
            bot.name = bot_prefix_ + std::to_string(id);
            bot.cloud_topic = "/" + bot.name + "/" + bot_cloud_topic_;
            bot.world_frame = bot.name + "/world";
            bot.sensor_frame = bot.name + "/" + bot_sensor_frame_;
            bot.forced_cloud_frame = bot_cloud_frame_.empty() ? std::string{}:(bot.name + "/" + bot_cloud_frame_);

            bots_.emplace(id, bot);

            RCLCPP_INFO(get_logger(),
                        "Configured bot %d: cloud_topic=%s, sensor_frame=%s",
                        id, bot.cloud_topic.c_str(), bot.sensor_frame.c_str());
        }
    }

    void EsdfMapNode::setupSubscriptions() {
        if (input_mode_ == "cloud") {
            cloud_sub_ = create_subscription<PointCloud2>(
                cloud_topic_,
                rclcpp::SensorDataQoS(),
                [this](PointCloud2::SharedPtr msg) { handleCloud(msg); });
            return;
        }

        // robots mode
        for (auto &kv : bots_) {
            auto &bot = kv.second;
            bot.sub = create_subscription<PointCloud2>(
                bot.cloud_topic,
                rclcpp::SensorDataQoS(),
                [this, bot_id = bot.id](PointCloud2::SharedPtr msg) {
                    handleBotCloud(bot_id, msg);
                });
        }
    }

    void EsdfMapNode::setupTimers() {
        auto clock = this->get_clock();  // ROS / sim time clock
        auto node_base = this->get_node_base_interface();
        auto timer_iface = this->get_node_timers_interface();

        if (!integrate_every_cloud_ && esdf_update_rate_hz_ > 0.0) {
            auto period = rclcpp::Duration::from_seconds(1.0 / esdf_update_rate_hz_);
            esdf_update_timer_ = rclcpp::create_timer(
                node_base, timer_iface, clock, period,
                std::bind(&EsdfMapNode::esdfUpdateTimerCb, this));
        }

        if (publish_rate_hz_ > 0.0) {
            auto period = rclcpp::Duration::from_seconds(1.0 / publish_rate_hz_);
            publish_timer_ = rclcpp::create_timer(
                node_base, timer_iface, clock, period,
                std::bind(&EsdfMapNode::publishTimerCb, this));
        }

        if (debug_profile_ && !heartbeat_timer_) {
            auto period = rclcpp::Duration::from_seconds(2.0);
            heartbeat_timer_ = rclcpp::create_timer(
                node_base, timer_iface, clock, period,
                [this]() {
                    RCLCPP_INFO(get_logger(),
                        "[profile] mode=%s core=%s publish_full=%d roi=%d cost2d=%d",
                        input_mode_.c_str(),
                        core_ ? "ok" : "null",
                        publish_full_grid_, publish_roi_grid_, publish_costmap_2d_);
                });
        }
    }

    void EsdfMapNode::setupService() {
        query_srv_ = create_service<EsdfQuery>( "esdf/query",
            std::bind(&EsdfMapNode::handleQuery, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

    bool EsdfMapNode::transformCloud(const PointCloud2& in, PointCloud2& out,
        const std::string& in_frame, const std::string& out_frame)
    {
         if (!tf_buffer_) return false;

        if (in_frame.empty()) {
            RCLCPP_WARN(get_logger(), "Incoming cloud has empty frame_id.");
            return false;
        }

        if (in_frame == out_frame) {
            out = in;
            out.header.frame_id = out_frame;
            return true;
        }

        const auto clock_type = this->get_clock()->get_clock_type();
        rclcpp::Time stamp(in.header.stamp, clock_type);

        // If stamp is zero, request "latest" transform
        if (stamp.nanoseconds() == 0) {
            stamp = rclcpp::Time(0, 0, clock_type);  // latest
        }

        try {
            auto tf = tf_buffer_->lookupTransform(
                out_frame, in_frame, stamp,
                rclcpp::Duration::from_seconds(tf_timeout_sec_));

            tf2::doTransform(in, out, tf);
            out.header.stamp = in.header.stamp;
            out.header.frame_id = out_frame;
            return true;

        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "TF cloud transform failed (%s -> %s): %s",
                in_frame.c_str(), out_frame.c_str(), ex.what());
            return false;
        }
    }

    void EsdfMapNode::handleCloudMsg(const PointCloud2::SharedPtr& msg, const std::string& source_frame) {
        if (!core_ || !msg) return;

        const auto t_integrate_start = std::chrono::steady_clock::now();

        PointCloud2 cloud_world;
        const PointCloud2* src = msg.get();

        if (source_frame != world_frame_) {
            if (!transformCloud(*msg, cloud_world, source_frame, world_frame_)) return;
            src = &cloud_world;
        }
        

        PointCloud cloud_M;
        pointCloud2ToPcl(*src, cloud_M);

        Eigen::Isometry3d T_M_L = Eigen::Isometry3d::Identity(); // TODO: currently unused in w\o rayCast
        core_->integrateCloud(cloud_M, T_M_L);
        const auto dt_integrate = std::chrono::steady_clock::now() - t_integrate_start;
        recordTiming("1. integrate_cloud", dt_integrate);

        if (integrate_every_cloud_) {
            const auto t_update_start = std::chrono::steady_clock::now();
            core_->updateEsdf();
            const auto dt_update = std::chrono::steady_clock::now() - t_update_start;
            recordTiming("2. update_esdf", dt_update);

            if (debug_profile_) {
                auto count = ++debug_profile_counter_;
                if (debug_profile_every_ > 0 && (count % static_cast<std::uint64_t>(debug_profile_every_) == 0)) {
                    const double ms_int = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt_integrate).count();
                    const double ms_upd = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt_update).count();
                    RCLCPP_INFO(get_logger(),
                        "[profile] cloud %zu pts=%zu tf_src=%s->%s integrate=%.2fms update=%.2fms",
                        static_cast<size_t>(count),
                        cloud_M.size(),
                        source_frame.c_str(), world_frame_.c_str(),
                        ms_int, ms_upd);
                }
            }

        } else if (debug_profile_) {
            auto count = ++debug_profile_counter_;
            if (debug_profile_every_ > 0 && (count % static_cast<std::uint64_t>(debug_profile_every_) == 0)) {
                const double ms_int = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt_integrate).count();
                RCLCPP_INFO(get_logger(),
                    "[profile] cloud %zu pts=%zu tf_src=%s->%s integrate=%.2fms (ESDF deferred)",
                    static_cast<size_t>(count),
                    cloud_M.size(),
                    source_frame.c_str(), world_frame_.c_str(),
                    ms_int);
            }
        }
    
    }

    void EsdfMapNode::handleCloud(const PointCloud2::SharedPtr msg) {
        if (!msg) return;
        std::string src_frame = cloud_frame_.empty() ? msg->header.frame_id : cloud_frame_;
        handleCloudMsg(msg, src_frame);
    }

    void EsdfMapNode::handleBotCloud(int bot_id, const PointCloud2::SharedPtr msg) {
        auto it = bots_.find(bot_id);
        if (it == bots_.end()) return;
        const std::string& src_frame = it->second.forced_cloud_frame.empty() ? msg->header.frame_id : it->second.forced_cloud_frame;
        handleCloudMsg(msg, src_frame);
    }

    bool EsdfMapNode::lookupLidarPose(const std::string &lidar_frame,
                                      const rclcpp::Time &stamp,
                                      Eigen::Isometry3d &T_M_L)
    {
        try {
            auto tf = tf_buffer_->lookupTransform(
                world_frame_, lidar_frame, stamp,
                rclcpp::Duration::from_seconds(tf_timeout_sec_));

            const auto &tr = tf.transform.translation;
            const auto &qr = tf.transform.rotation;
            Eigen::Quaterniond q(qr.w, qr.x, qr.y, qr.z);
            Vec3d t(tr.x, tr.y, tr.z);

            T_M_L = Eigen::Isometry3d::Identity();
            T_M_L.linear() = q.toRotationMatrix();
            T_M_L.translation() = t;
            return true;
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "TF lookup failed (%s -> %s): %s",
                                 world_frame_.c_str(), lidar_frame.c_str(), ex.what());
            return false;
        }
    }

    void EsdfMapNode::esdfUpdateTimerCb() {
        if (!core_) return;
        const auto t_start = std::chrono::steady_clock::now();
        core_->updateEsdf();
        const auto dt = std::chrono::steady_clock::now() - t_start;
        recordTiming("2. update_esdf", dt);

        if (debug_profile_) {
            const double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dt).count();
            RCLCPP_INFO(get_logger(), "[profile] timer update_esdf=%.2fms", ms);
        }
    }

    void EsdfMapNode::publishTimerCb() {
        if (!core_) return;

        if (publish_full_grid_) publishGrid(true); // full region
        if (publish_roi_grid_)  publishGrid(false);  // ROI (consumeUpdateRegion)
        if (publish_costmap_2d_) publishCostmap2D();
        if (time_log_) logTimingReport();
        if (debug_profile_) {
            RCLCPP_INFO(get_logger(), "[profile] publish tick full=%d roi=%d cost2d=%d",
                        publish_full_grid_, publish_roi_grid_, publish_costmap_2d_);
        }
    }

    static void initCloudFields(sensor_msgs::msg::PointCloud2& msg) {
        sensor_msgs::PointCloud2Modifier init(msg);
        init.setPointCloud2Fields(
            4,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32
        );
    }

    void EsdfMapNode::publishGrid(bool full_region) {
        auto pub = full_region ? esdf_grid_pub_ : esdf_grid_roi_pub_;
        if (!pub) return;

        // Use the persistent message storage but re-init the fields each time to
        // guarantee the layout/point_step stays consistent even if previous calls
        // exited early.
        sensor_msgs::msg::PointCloud2& msg = full_region ? msg_full_ : msg_roi_;
        initCloudFields(msg);
        UpdateRegion roi;
        if (!full_region) {
            roi = core_->consumeUpdateRegion();
            if (!roi.valid) return;
        }

        const auto t_get = std::chrono::steady_clock::now();

        std::vector<esdf_map::VoxelSample> filtered;
        filtered.reserve(1024 * 64);
        {
            auto locked = full_region ? core_->getAllLockedVoxels()
                                    : core_->getLockedVoxelView(roi);
            const auto& view = locked.view();
            if (view.count() == 0) return;

            const float max_d = static_cast<float>(grid_max_distance_);
            view.forEach([&](const esdf_map::VoxelSample& v) {
                if (v.distance <= max_d) filtered.push_back(v);
            });
        }
        if (filtered.empty()) return;

        recordTiming(full_region ? "get_voxels" : "get_voxels_roi", std::chrono::steady_clock::now() - t_get);

        const auto t_pub = std::chrono::steady_clock::now();
        const std::size_t N = filtered.size();
        const uint32_t point_step = static_cast<uint32_t>(4 * sizeof(float));
        msg.point_step = point_step;
        msg.height = 1;
        msg.width = static_cast<uint32_t>(N);
        msg.row_step = point_step * msg.width;
        msg.is_dense = false;
        msg.data.assign(static_cast<size_t>(msg.row_step) * msg.height, 0u);

        float *data_ptr = reinterpret_cast<float *>(msg.data.data());
        for (size_t i = 0; i < N; ++i) {
            const auto& v = filtered[i];
            data_ptr[i * 4 + 0] = v.x;
            data_ptr[i * 4 + 1] = v.y;
            data_ptr[i * 4 + 2] = v.z;
            data_ptr[i * 4 + 3] = v.distance;
        }

        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = world_frame_;
        pub->publish(msg);

        recordTiming(full_region ? "publish_esdf" : "publish_esdf_roi", std::chrono::steady_clock::now() - t_pub);
    }


    void EsdfMapNode::publishCostmap2D() {
        const auto t_total_start = std::chrono::steady_clock::now();
        if (!costmap_pub_) return;

        Slice2D slice;
        core_->extractSlice(costmap_layer_z_, slice);

        if (slice.dims.x() == 0 || slice.dims.y() == 0) return;

        OccupancyGrid msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = world_frame_;
        fillCostmapMsg(slice, msg);
        costmap_pub_->publish(msg);
        recordTiming("publish_costmap", std::chrono::steady_clock::now() - t_total_start);
    }

    void EsdfMapNode::pointCloud2ToPcl(const PointCloud2 &msg,
                                       PointCloud &cloud_out) const
    {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, cloud_out);
    }

    void EsdfMapNode::fillCostmapMsg(const Slice2D &slice,
                                     OccupancyGrid &grid_msg) const
    {
        grid_msg.info.resolution = static_cast<float>(slice.resolution);
        grid_msg.info.width = static_cast<uint32_t>(slice.dims.x());
        grid_msg.info.height = static_cast<uint32_t>(slice.dims.y());
        grid_msg.info.origin.position.x = slice.origin.x();
        grid_msg.info.origin.position.y = slice.origin.y();
        grid_msg.info.origin.position.z = costmap_layer_z_;
        grid_msg.info.origin.orientation.w = 1.0;
        grid_msg.info.origin.orientation.x = 0.0;
        grid_msg.info.origin.orientation.y = 0.0;
        grid_msg.info.origin.orientation.z = 0.0;

        const int nx = slice.dims.x();
        const int ny = slice.dims.y();

        grid_msg.data.resize(static_cast<size_t>(nx * ny));

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx = y * nx + x;
                float d = slice.distances[idx];

                int8_t cost = 0;
                if (d >= static_cast<float>(costmap_free_distance_)) {
                    cost = 0;
                } else if (d <= static_cast<float>(costmap_lethal_distance_)) {
                    cost = 100;
                } else {
                    double t = (d - costmap_lethal_distance_) /
                               (costmap_free_distance_ - costmap_lethal_distance_);
                    t = std::clamp(t, 0.0, 1.0);
                    cost = static_cast<int8_t>(std::lround(100.0 * (1.0 - t)));
                }
                grid_msg.data[idx] = cost;
            }
        }
    }

    void EsdfMapNode::recordTiming(const std::string &name,
                                   std::chrono::steady_clock::duration duration)
    {
        if (!time_log_) return;
        const auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();

        auto &buffer = timing_history_[name];
        buffer.emplace_back(now, ms);

        // Drop samples older than window
        const auto window_start = now - timing_window_;
        while (!buffer.empty() && buffer.front().first < window_start) {
            buffer.pop_front();
        }

        (void)buffer;
    }

    bool EsdfMapNode::computeTimingStats(const std::string &name, double &curr, double &mean_ms,
                                         double &std_ms, double &max_ms,
                                         std::size_t &count)
    {
        auto it = timing_history_.find(name);
        if (it == timing_history_.end() || it->second.empty()) {
            return false;
        }

        const auto &buffer = it->second;
        double sum = 0.0;
        double sum_sq = 0.0;
        double max_val = 0.0;
        for (const auto &p : buffer) {
            sum += p.second;
            sum_sq += p.second * p.second;
            max_val = std::max(max_val, static_cast<double>(p.second));
        }
        curr = buffer[buffer.size()-1].second;
        count = buffer.size();
        mean_ms = sum / static_cast<double>(count);
        const double var = std::max(0.0, sum_sq / static_cast<double>(count) - mean_ms * mean_ms);
        std_ms = std::sqrt(var);
        max_ms = max_val;
        return true;
    }

    void EsdfMapNode::logTimingReport()
    {
        std::ostringstream oss;
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(timing_window_).count();
        oss << "[time " << secs << "s]\n";
        oss << "--------------------------------------------------------\n";
        oss << "  " << std::left << std::setw(21) << "Key"
            << std::right 
            << std::setw(6) << "Curr"
            << std::setw(6) << "Mean"
            << std::setw(6) << "Std"
            << std::setw(6) << "Max"
            << std::setw(5) << "#" << "  (ms)\n";
        oss << "--------------------------------------------------------\n";

        for (const auto &[k,_] : timing_history_) {
            double curr = 0.0, mean = 0.0, stddev = 0.0, maxv = 0.0;
            std::size_t n = 0;

            if (computeTimingStats(k, curr, mean, stddev, maxv, n)) {
                oss << "  " << std::left << std::setw(21) << k
                    << std::right << std::fixed 
                    << std::setw(6) << std::setprecision(1) << curr
                    << std::setw(6) << std::setprecision(1) << mean
                    << std::setw(6) << std::setprecision(1) << stddev
                    << std::setw(6) << std::setprecision(1) << maxv
                    << std::setw(5) << n << "\n";
            }
        }

        oss << "--------------------------------------------------------\n";
        RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());
    }

    void EsdfMapNode::handleQuery(const std::shared_ptr<EsdfQuery::Request> request,
                                  std::shared_ptr<EsdfQuery::Response> response)
    {
        response->success = false;
        response->observed = false;
        response->distance = 0.0f;
        response->gradient.x = 0.0;
        response->gradient.y = 0.0;
        response->gradient.z = 0.0;

        if (!core_) return;

        std::string frame = request->frame_id;
        if (frame.empty()) {
            frame = world_frame_;
        }

        Vec3d p_M;

        if (frame == world_frame_) {
            p_M = Vec3d(
                request->position.x,
                request->position.y,
                request->position.z);
        } else {
            try {
                auto tf = tf_buffer_->lookupTransform(
                    world_frame_, frame, rclcpp::Time(0),
                    rclcpp::Duration::from_seconds(tf_timeout_sec_));
                const auto &tr = tf.transform.translation;
                const auto &qr = tf.transform.rotation;
                Eigen::Quaterniond q(qr.w, qr.x, qr.y, qr.z);
                Vec3d t(tr.x, tr.y, tr.z);
                Vec3d p_local(
                    request->position.x,
                    request->position.y,
                    request->position.z);
                p_M = q * p_local + t;
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(get_logger(),
                            "ESDF query TF error (%s -> %s): %s",
                            world_frame_.c_str(), frame.c_str(), ex.what());
                return;
            }
        }

        double dist;
        Vec3d grad;
        if (!core_->queryDistanceAndGradient(p_M, dist, grad)) {
            return;
        }

        response->success = true;
        response->distance = static_cast<float>(dist);
        response->gradient.x = grad.x();
        response->gradient.y = grad.y();
        response->gradient.z = grad.z();
        response->observed = core_->isObserved(p_M);
    }

} // namespace esdf_map
