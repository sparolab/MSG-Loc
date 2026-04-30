#include <ros/ros.h>
#include <fstream>  // For file operations
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/map.hpp>
#include <gtsam/inference/Key.h>  // GTSAM keys
#include <gtsam/nonlinear/Values.h> // GTSAM Values
#include <gtsam/nonlinear/Symbol.h>  // GTSAM Symbol
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
//gtsam_quadric//
#include <gtsam_quadrics/geometry/ConstrainedDualQuadric.h>
#include <gtsam_quadrics/geometry/QuadricCamera.h>
#include <gtsam_quadrics/geometry/BoundingBoxFactor.h>
//opencv//
#include <opencv2/imgcodecs.hpp>   // cv::imread, cv::IMREAD_*
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>  // Include OpenCV for Eigen::Vector3d and Eigen::Vector4d
#include <opencv2/core/persistence.hpp>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include <algorithm>
using json = nlohmann::json;
///////////////////////////// method  //////////////////////////
using LabelCandidateMap = std::map<std::string, std::pair<std::vector<uint8_t>, float>>;

struct MsglocGlobalVariable {
    std::string config_path = "/root/workspace/src/msgloc/Config/config.yaml";

    // Input/output paths
    std::string map_path = "/root/workspace/map.json";
    std::string detection_path = "/root/workspace/detections/desk_tap_lvis_detections.json";
    std::string base_path = "/root/workspace/dataset_root/rgbd_dataset_freiburg2_desk";
    std::string cam_info_path = "/root/workspace/src/msgloc/Cameras/TUM2.yaml";
    std::string output_dir = "/root/workspace/msgloc_results";

    // Graph and matching parameters
    int kNeighbors = 5;   // Number of nearest neighbors for source-graph edge construction
    double minRangeM = 0.10;  // Minimum valid object range in meters
    double maxRangeM = 20.0;  // Maximum valid object range in meters
    double assocTolSec = 0.050;  // Timestamp association tolerance (seconds)
    int valid_label_Topk = 5;  // Number of top labels kept per detection
    int matchTopK = 3;  // Number of target candidates kept for each source node
    int minValidLabelCount = 4;  // Minimum matched labels required for a candidate pair
    int maxLabelMatches = 5;  // Maximum label matches evaluated per pair
    double matchDistanceThreshold = 1.0;  // Neighbor edge distance tolerance

    // Pose estimation parameters
    int RANSAC_iters = 2000;  // Maximum RANSAC iterations for pose estimation
    int ompNumThreads = 8;
    double wassersteinConstant = 100.0;
    double poseInlierThreshold = 0.5;

    // Offline sequence range
    int sequenceIndex = 0;
    int startFrame = 0;
    int endFrame = -1;

    // Camera parameters loaded from cam_info_path
    double width = 640, height = 480, f_x = 525.0, f_y = 525.0, c_x = 319.5, c_y = 239.5, skew = 0.0, depth_factor = 5208.0;
    double d_k1 = 0.0, d_k2 = 0.0, d_p1 = 0.0, d_p2 = 0.0, d_k3 = 0.0;

    std::map<gtsam::Key, LabelCandidateMap> SourceLabelCandidates;  // Per-source-key label candidates and scores

    void loadFromYaml(const std::string& yaml_path) {
        cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open config yaml file: " + yaml_path);
        }
        config_path = yaml_path;

        auto read_string = [&fs](const char* key, std::string& out) {
            cv::FileNode node = fs[key];
            if (!node.empty()) out = static_cast<std::string>(node);
        };
        auto read_int = [&fs](const char* key, int& out) {
            cv::FileNode node = fs[key];
            if (!node.empty()) out = static_cast<int>(node);
        };
        auto read_double = [&fs](const char* key, double& out) {
            cv::FileNode node = fs[key];
            if (!node.empty()) out = static_cast<double>(node);
        };

        read_string("map_path", map_path);
        read_string("detection_path", detection_path);
        read_string("base_path", base_path);
        read_string("cam_info", cam_info_path);
        read_string("cam_info_path", cam_info_path);
        read_string("output_dir", output_dir);

        read_int("k_neighbors", kNeighbors);
        read_int("kNeighbors", kNeighbors);
        read_double("min_range_m", minRangeM);
        read_double("minRangeM", minRangeM);
        read_double("max_range_m", maxRangeM);
        read_double("maxRangeM", maxRangeM);
        read_double("assoc_tol_sec", assocTolSec);
        read_double("assocTolSec", assocTolSec);
        read_int("valid_label_topk", valid_label_Topk);
        read_int("valid_label_Topk", valid_label_Topk);
        read_int("match_topk", matchTopK);
        read_int("min_valid_label_count", minValidLabelCount);
        read_int("max_label_matches", maxLabelMatches);
        read_double("match_distance_threshold", matchDistanceThreshold);

        read_int("ransac_iters", RANSAC_iters);
        read_int("RANSAC_iters", RANSAC_iters);
        read_int("omp_num_threads", ompNumThreads);
        read_double("wasserstein_constant", wassersteinConstant);
        read_double("pose_inlier_threshold", poseInlierThreshold);

        read_int("sequence_index", sequenceIndex);
        read_int("start_frame", startFrame);
        read_int("end_frame", endFrame);

        if (kNeighbors < 0) throw std::runtime_error("config error: k_neighbors must be >= 0");
        if (minRangeM < 0.0) throw std::runtime_error("config error: min_range_m must be >= 0");
        if (maxRangeM < minRangeM) throw std::runtime_error("config error: max_range_m must be >= min_range_m");
        if (valid_label_Topk <= 0) throw std::runtime_error("config error: valid_label_topk must be > 0");
        if (matchTopK <= 0) throw std::runtime_error("config error: match_topk must be > 0");
        if (minValidLabelCount <= 0) throw std::runtime_error("config error: min_valid_label_count must be > 0");
        if (maxLabelMatches <= 0) throw std::runtime_error("config error: max_label_matches must be > 0");
        if (RANSAC_iters <= 0) throw std::runtime_error("config error: ransac_iters must be > 0");
        if (ompNumThreads <= 0) throw std::runtime_error("config error: omp_num_threads must be > 0");
        if (wassersteinConstant <= 0.0) throw std::runtime_error("config error: wasserstein_constant must be > 0");
        if (poseInlierThreshold < 0.0) throw std::runtime_error("config error: pose_inlier_threshold must be >= 0");
        if (sequenceIndex < 0) throw std::runtime_error("config error: sequence_index must be >= 0");
        if (minValidLabelCount > valid_label_Topk) {
            ROS_WARN("min_valid_label_count (%d) is greater than valid_label_topk (%d). Matching may reject every candidate.",
                     minValidLabelCount, valid_label_Topk);
        }
    }

    void loadRosPathParams(const ros::NodeHandle& pnh) {
        pnh.param<std::string>("map_path", map_path, map_path);
        pnh.param<std::string>("detection_path", detection_path, detection_path);
        pnh.param<std::string>("base_path", base_path, base_path);
        pnh.param<std::string>("cam_info", cam_info_path, cam_info_path);
        pnh.param<std::string>("cam_info_path", cam_info_path, cam_info_path);
        pnh.param<std::string>("output_dir", output_dir, output_dir);
    }
};

static MsglocGlobalVariable g_msgloc;

// Define QuadricGraph class
struct QuadricNode {
    Eigen::Vector3d translation;
    Eigen::Vector4d orientation;  // Updated to store quaternion
    Eigen::Vector3d radii;
    std::vector<gtsam::Key> neighbor_node;
    gtsam_quadrics::AlignedBox2 bound;
    int sequence;  // Sequence number of the node (odometry frame number)
        // Serialization function
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        // Serialize Eigen::Vector3d (translation)
        ar & translation.x() & translation.y() & translation.z();

        // Serialize Eigen::Vector4d (orientation, quaternion)
        ar & orientation.w() & orientation.x() & orientation.y() & orientation.z();

        // Serialize Eigen::Vector3d (radii)
        ar & radii.x() & radii.y() & radii.z();

        // Serialize std::vector (neighbor_node)
        ar & neighbor_node;

    }
};

// ---------------------- headers/globals ----------------------
struct StageTimer {
    double sum_all   = 0, sum_succ = 0, sum_insuf = 0, sum_fail = 0;
    int    cnt_all   = 0, cnt_succ = 0, cnt_insuf = 0, cnt_fail = 0;

    inline void add(double t, int sign) {
        sum_all += t; ++cnt_all;
        if (sign == 0)      { sum_succ += t; ++cnt_succ; }
        else if (sign == 1) { sum_insuf += t; ++cnt_insuf; }
        else                { sum_fail += t; ++cnt_fail; }
    }
    inline double mean_all()   const { return cnt_all   ? sum_all   / cnt_all   : 0.0; }
    inline double mean_succ()  const { return cnt_succ  ? sum_succ  / cnt_succ  : 0.0; }
    inline double mean_insuf() const { return cnt_insuf ? sum_insuf / cnt_insuf : 0.0; }
    inline double mean_fail()  const { return cnt_fail  ? sum_fail  / cnt_fail  : 0.0; }
};

// Stage timers per phase (preferred over legacy sum/mean variables)
StageTimer T_graph, T_match, T_pose, T_total;


class QuadricGraph {
public:
    std::map<gtsam::Key, QuadricNode> nodes;
    // std::vector<std::tuple<gtsam::Key, gtsam::Key, double, double>> edges;  // Stores distance and cosine similarity together
    std::map<std::pair<gtsam::Key,gtsam::Key>, double> edges; 

    // Add a new node
    void addNode(gtsam::Key key, const QuadricNode& node) {
        nodes[key] = node;
    }

    // Add an edge
    void addEdge(gtsam::Key key1, gtsam::Key key2, double distance) {
        if (key1 > key2){
            std::swap(key1,key2); // Keep the smaller key first
        }
        if (!has_edge(key1,key2)){
            // double dis_weight = 1.0 / (1.0 + distance);
            edges[std::make_pair(key1,key2)] = distance;
            nodes[key1].neighbor_node.push_back(key2);
            nodes[key2].neighbor_node.push_back(key1);
        }
    }
	
    bool has_edge(gtsam::Key key1, gtsam::Key key2) const {	                
        return edges.count(std::make_pair(key1,key2)) > 0;
    }


    // Serialization function
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & nodes;
        ar & edges;
    }
};

//// Storage class definition ////
struct QuadricData {
    QuadricGraph graph;
    std::map<gtsam::Key, std::map<std::string, std::vector<float>>> TargetLabelLikelihood;
    // [Added] BaselineLabel: { p_norm, sum_label, total_sum }
    std::map<gtsam::Key, std::map<std::string, std::vector<float>>> BaselineLabel;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & graph;
        ar & TargetLabelLikelihood;
        ar & BaselineLabel;

                // [Added] Save/load order must exactly match the writer side

    }
};

class StorageManager {
public:
    // Load all data from a single file
    static void loadAllData(QuadricData& data, const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            ROS_ERROR("Failed to open file for loading quadric data.");
            return;
        }

        json j;
        try {
            ifs >> j;
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to parse map json: %s", e.what());
            return;
        }

        if (!j.is_object()) {
            ROS_ERROR("Invalid map json root (must be object).");
            return;
        }

        if (!j.contains("nodes") || !j["nodes"].is_array()) {
            ROS_ERROR("Invalid map json: 'nodes' array not found.");
            return;
        }

        data = QuadricData{};  // clear previous state

        auto parse_key = [](const json& v, gtsam::Key& out) -> bool {
            try {
                if (v.is_number_unsigned()) {
                    out = static_cast<gtsam::Key>(v.get<uint64_t>());
                    return true;
                }
                if (v.is_number_integer()) {
                    const long long n = v.get<long long>();
                    if (n < 0) return false;
                    out = static_cast<gtsam::Key>(static_cast<uint64_t>(n));
                    return true;
                }
                if (v.is_string()) {
                    const std::string s = v.get<std::string>();
                    out = static_cast<gtsam::Key>(std::stoull(s));
                    return true;
                }
            } catch (...) {
                return false;
            }
            return false;
        };

        auto parse_vec3 = [](const json& arr, Eigen::Vector3d& out) -> bool {
            if (!arr.is_array() || arr.size() < 3) return false;
            try {
                out = Eigen::Vector3d(arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>());
                return true;
            } catch (...) {
                return false;
            }
        };

        auto parse_vec4 = [](const json& arr, Eigen::Vector4d& out) -> bool {
            if (!arr.is_array() || arr.size() < 4) return false;
            try {
                out = Eigen::Vector4d(arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>(), arr[3].get<double>());
                return true;
            } catch (...) {
                return false;
            }
        };

        auto parse_label_map = [](const json& obj, std::map<std::string, std::vector<float>>& out) {
            if (!obj.is_object()) return;
            for (auto it = obj.begin(); it != obj.end(); ++it) {
                const std::string label = it.key();
                const json& val = it.value();
                std::vector<float> vec;
                if (val.is_array()) {
                    vec.reserve(val.size());
                    for (const auto& x : val) {
                        if (x.is_number()) {
                            vec.push_back(static_cast<float>(x.get<double>()));
                        } else if (x.is_array()) {
                            for (const auto& y : x) {
                                if (y.is_number()) vec.push_back(static_cast<float>(y.get<double>()));
                            }
                        }
                    }
                } else if (val.is_number()) {
                    vec.push_back(static_cast<float>(val.get<double>()));
                }
                if (!vec.empty()) out[label] = std::move(vec);
            }
        };

        // nodes
        for (const auto& n : j["nodes"]) {
            if (!n.is_object() || !n.contains("id")) continue;

            gtsam::Key key = 0;
            if (!parse_key(n["id"], key)) {
                ROS_WARN("Skipping node with invalid id in map json.");
                continue;
            }

            QuadricNode node;
            node.translation = Eigen::Vector3d::Zero();
            node.orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);  // [w, x, y, z]
            node.radii = Eigen::Vector3d::Zero();
            node.sequence = n.value("sequence", 0);

            if (n.contains("translation")) parse_vec3(n["translation"], node.translation);
            if (n.contains("orientation")) parse_vec4(n["orientation"], node.orientation);
            if (n.contains("radii")) parse_vec3(n["radii"], node.radii);

            if (n.contains("neighbor_node") && n["neighbor_node"].is_array()) {
                for (const auto& k : n["neighbor_node"]) {
                    gtsam::Key nk = 0;
                    if (parse_key(k, nk)) node.neighbor_node.push_back(nk);
                }
            }

            data.graph.addNode(key, node);

            // Python storage key: msgloc_label
            if (n.contains("msgloc_label")) {
                parse_label_map(n["msgloc_label"], data.TargetLabelLikelihood[key]);
            }
            // Compatibility key: target_label_likelihood
            else if (n.contains("target_label_likelihood")) {
                parse_label_map(n["target_label_likelihood"], data.TargetLabelLikelihood[key]);
            }

            if (n.contains("baseline_label")) {
                parse_label_map(n["baseline_label"], data.BaselineLabel[key]);
            }
        }

        // edges
        if (j.contains("edges") && j["edges"].is_array()) {
            for (const auto& e : j["edges"]) {
                if (!e.is_object() || !e.contains("from") || !e.contains("to")) continue;
                gtsam::Key k1 = 0, k2 = 0;
                if (!parse_key(e["from"], k1) || !parse_key(e["to"], k2)) continue;
                const double dist = e.value("distance", 0.0);

                // Safety: avoid implicit default-node creation for missing-node edges
                if (data.graph.nodes.find(k1) == data.graph.nodes.end() ||
                    data.graph.nodes.find(k2) == data.graph.nodes.end()) {
                    ROS_WARN("Skipping edge with missing node(s): %lu - %lu",
                             gtsam::Symbol(k1).index(), gtsam::Symbol(k2).index());
                    continue;
                }
                data.graph.addEdge(k1, k2, dist);
            }
        }

        ifs.close();
        ROS_INFO("Map json loaded successfully from %s. nodes=%zu edges=%zu labels=%zu baseline=%zu",
                 filepath.c_str(),
                 data.graph.nodes.size(),
                 data.graph.edges.size(),
                 data.TargetLabelLikelihood.size(),
                 data.BaselineLabel.size());
    }

};

/////////////////////////////// End of database handling ///////////////////////////////////////
class Msgloc {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher map_object_marker_pub_;
    ros::Publisher map_edge_marker_pub_;
    ros::Publisher pose_marker_pub_;
    ros::Publisher pose_odom_pub_;
    std::string rviz_frame_id_ = "test";
    std::string pose_child_frame_id_ = "msgloc_camera";
    std::vector<visualization_msgs::Marker> pose_arrow_markers_;
    std::vector<geometry_msgs::Point> pose_path_points_;
    int pose_marker_id_ = 0;

    static std::string joinPath(const std::string& a, const std::string& b) {
        if (a.empty()) return b;
        if (b.empty()) return a;
        const char sep =
        #ifdef _WIN32
            '\\';
        #else
            '/';
        #endif
        if (a.back() == sep) return a + b;
        return a + sep + b;
    }
    std::string resolvePath(const std::string& rel) const {
        if (rel.empty()) return rel;
        // Keep absolute paths as-is
        if (rel.front() == '/'
        #ifdef _WIN32
            || (rel.size()>1 && rel[1]==':')
        #endif
        ) return rel;
        if (g_msgloc.base_path.empty()) return rel;
        return joinPath(g_msgloc.base_path, rel);
    }

    geometry_msgs::Point makePoint(double x, double y, double z) const {
        geometry_msgs::Point p;
        p.x = x;
        p.y = y;
        p.z = z;
        return p;
    }

    visualization_msgs::Marker makeDeleteAllMarker(const ros::Time& stamp) const {
        visualization_msgs::Marker marker;
        marker.header.frame_id = rviz_frame_id_;
        marker.header.stamp = stamp;
        marker.action = visualization_msgs::Marker::DELETEALL;
        return marker;
    }

    double markerScaleFromRadius(double r) const {
        if (!std::isfinite(r) || r <= 0.0) return 0.1;
        return std::max(0.1, 2.0 * r);
    }

    bool applyBaselineColor(gtsam::Key key, visualization_msgs::Marker& marker) const {
        const auto baseline_it = loadedData.BaselineLabel.find(key);
        if (baseline_it == loadedData.BaselineLabel.end()) return false;

        const auto& labels = baseline_it->second;
        if (labels.empty()) return false;

        const std::vector<float>* best_stats = nullptr;
        double best_p_norm = -1.0;
        for (const auto& kv : labels) {
            const auto& stats = kv.second;
            if (stats.empty()) continue;
            if (stats[0] > best_p_norm) {
                best_p_norm = stats[0];
                best_stats = &stats;
            }
        }

        if (!best_stats) return false;

        // map.json compact format: [p_norm, [r,g,b]] -> parsed as [p_norm,r,g,b].
        // Older in-memory format may be [p_norm,sum,total,r,g,b].
        size_t color_idx = 0;
        if (best_stats->size() == 4) {
            color_idx = 1;
        } else if (best_stats->size() >= 6) {
            color_idx = 3;
        } else {
            return false;
        }

        const double r = (*best_stats)[color_idx + 0];
        const double g = (*best_stats)[color_idx + 1];
        const double b = (*best_stats)[color_idx + 2];
        const double max_rgb = std::max({r, g, b});
        const double denom = (max_rgb > 1.0) ? 255.0 : 1.0;

        marker.color.r = std::clamp(r / denom, 0.0, 1.0);
        marker.color.g = std::clamp(g / denom, 0.0, 1.0);
        marker.color.b = std::clamp(b / denom, 0.0, 1.0);
        marker.color.a = 0.75;
        return true;
    }

    void setupVisualizationPublishers() {
        pnh_.param<std::string>("rviz_frame_id", rviz_frame_id_, rviz_frame_id_);
        pnh_.param<std::string>("pose_child_frame_id", pose_child_frame_id_, pose_child_frame_id_);

        map_object_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/msgloc_map_objects", 1, true);
        map_edge_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/msgloc_map_edges", 1, true);
        pose_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/msgloc_pose_markers", 1, true);
        pose_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/msgloc_pose_odom", 50);
    }

public:
    QuadricData loadedData;
    json all_json;
    int Pose_id = 0;
    gtsam::Values result_values;  // You need to define initialEstimate
    image_geometry::PinholeCameraModel cam_model;

    int save_count = 0;

    int Success_Count = 0;        // Success count
    int Insufficient_Count = 0;   // Count of insufficient unique source nodes
    int Failure_Count = 0;        // Failure count
    int Total_Count = 0;          // Total attempt count

    // === camera & pose for projection ===
    boost::shared_ptr<gtsam::Cal3_S2> calibration; // Required by QuadricCamera::project
    void loadCameraInfoFromYaml(const std::string& camera_yaml_path) {
        cv::FileStorage fs(camera_yaml_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            throw std::runtime_error("Failed to open camera yaml file: " + camera_yaml_path);
        }

        auto read_required_double = [&fs, &camera_yaml_path](const char* key, double& out) {
            cv::FileNode node = fs[key];
            if (node.empty()) {
                throw std::runtime_error(std::string("Missing required key in camera yaml (") + camera_yaml_path + "): " + key);
            }
            out = static_cast<double>(node);
        };

        read_required_double("Camera.fx", g_msgloc.f_x);
        read_required_double("Camera.fy", g_msgloc.f_y);
        read_required_double("Camera.cx", g_msgloc.c_x);
        read_required_double("Camera.cy", g_msgloc.c_y);
        read_required_double("Camera.width", g_msgloc.width);
        read_required_double("Camera.height", g_msgloc.height);
        read_required_double("DepthMapFactor", g_msgloc.depth_factor);

        // Distortion coefficients are optional; use 0.0 if absent.
        auto read_optional_double = [&fs](const char* key, double& out) {
            cv::FileNode node = fs[key];
            if (!node.empty()) out = static_cast<double>(node);
        };
        read_optional_double("Camera.k1", g_msgloc.d_k1);
        read_optional_double("Camera.k2", g_msgloc.d_k2);
        read_optional_double("Camera.p1", g_msgloc.d_p1);
        read_optional_double("Camera.p2", g_msgloc.d_p2);
        read_optional_double("Camera.k3", g_msgloc.d_k3);
        read_optional_double("Camera.skew", g_msgloc.skew);
    }

    Msgloc() : nh_(), pnh_("~")
    {
        setupVisualizationPublishers();
        loadCameraInfoFromYaml(g_msgloc.cam_info_path);

        ROS_INFO("Config path: %s", g_msgloc.config_path.c_str());
        ROS_INFO("Input map_path: %s", g_msgloc.map_path.c_str());
        ROS_INFO("Input json_path: %s", g_msgloc.detection_path.c_str());
        ROS_INFO("Input base_path: %s", g_msgloc.base_path.c_str());
        ROS_INFO("Input cam_info: %s", g_msgloc.cam_info_path.c_str());
        ROS_INFO("Output dir: %s", g_msgloc.output_dir.c_str());

        // 1) Load quadric map
        StorageManager::loadAllData(loadedData, g_msgloc.map_path);
        updateResultValues();
        setupCameraInfo();
        publishMapMarkers();
        // 2) Load detection JSON
        loadjson();
    }

// Set context independent of sequence metadata
void setSequenceContext(const json& sequence_obj) {
    (void)sequence_obj;
    // The base folder is controlled only by config base_path.
    ROS_INFO("Base folder set from config base_path: %s", g_msgloc.base_path.c_str());
}

static std::map<std::string, std::pair<std::vector<uint8_t>, float>>
normalize_label_map(const std::map<std::string, std::pair<std::vector<uint8_t>, float>>& label_map)
{
    std::map<std::string, std::pair<std::vector<uint8_t>, float>> norm_map;
    if (label_map.empty()) return norm_map;

    // Configure top-k locally in this function
    const std::size_t TOP_K = g_msgloc.valid_label_Topk;  // Change this value only if needed

    struct Item {
        std::string concept;
        std::vector<uint8_t> rgb;
        float score;  // Score clamped to non-negative range
    };

    // 1) Clamp negative scores to 0 and collect
    std::vector<Item> items;
    items.reserve(label_map.size());
    for (const auto& kv : label_map) {
        float s = kv.second.second;
        float s_clamped = std::max(0.0f, s);
        items.push_back(Item{kv.first, kv.second.first, s_clamped});
    }

    // 2) Sort by score descending (ties: concept lexicographical ascending)
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.concept < b.concept;
              });

    // 3) Select top-K
    const std::size_t k = std::min(TOP_K, items.size());

    // 4) Compute sum of top-K scores
    double sum_topk = 0.0;
    for (std::size_t i = 0; i < k; ++i) sum_topk += static_cast<double>(items[i].score);

    // 5) If the sum is 0, return uniform distribution over top-K
    if (sum_topk <= 0.0) {
        const float uniform = 1.0f / static_cast<float>(k);
        for (std::size_t i = 0; i < k; ++i) {
            // Return only top-K entries in the map
            norm_map.emplace(std::move(items[i].concept),
                             std::make_pair(std::move(items[i].rgb), uniform));
        }
        return norm_map;
    }

    // 6) L1-normalize and return only top-K
    for (std::size_t i = 0; i < k; ++i) {
        float s_norm = static_cast<float>(items[i].score / sum_topk);
        norm_map.emplace(std::move(items[i].concept),
                         std::make_pair(std::move(items[i].rgb), s_norm));
    }
    return norm_map;
}

/////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

std::tuple<QuadricGraph, double>
build_source_graph_from_frame(const json& frame) {
    QuadricGraph source_graph;

    // --- Timestamp ---
    const double stamp = frame.value("stamp", 0.0);

    // --- detections ---
    const bool has_dets = frame.contains("detections") && frame["detections"].is_array();
    if (!has_dets) {
        return std::make_tuple(source_graph, stamp);
    }

    // Optional: use provided sequence index (odometry frame id) if present
    const int sequence_idx = frame.value("sequence", 0);

    // Each detection -> node
    const auto& dets = frame["detections"];
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        const auto& det = dets[i];

        // bbox_xywh → [xmin,ymin,xmax,ymax]
        const auto& b = det.at("bbox_xywh");
        const double xc = b[0].get<double>(), yc = b[1].get<double>();
        const double w  = b[2].get<double>(), h  = b[3].get<double>();
        const double xmin = xc - 0.5*w, xmax = xc + 0.5*w;
        const double ymin = yc - 0.5*h, ymax = yc + 0.5*h;
        gtsam_quadrics::AlignedBox2 qbox(xmin, ymin, xmax, ymax);

        // --- depth median: directly use detection JSON field ---
        double md = -1.0;
        if (det.contains("median_depth") && det["median_depth"].is_number()) {
            md = det["median_depth"].get<double>();
        }
        if (!std::isfinite(md) || md <= 0.0) {
            continue;
        }

        // --- 3D center computation ---
        Eigen::Vector3d center(0,0,0);
        if (md > 0.0) {
            const cv::Point3d ray = cam_model.projectPixelTo3dRay(cv::Point2d(xc, yc)); // Unit vector
            center = Eigen::Vector3d(ray.x, ray.y, ray.z) * md; // md is in meters from detection JSON
        }
        //std::cout << "  [build_source_graph_from_frame] center: " << center.transpose() << std::endl;
        // --- Build node ---
        QuadricNode node;
        node.translation = center;
        node.orientation = Eigen::Vector4d(1,0,0,0); // Update if needed
        node.radii       = Eigen::Vector3d(0,0,0);   // Update if needed
        node.bound       = qbox;
        node.sequence    = sequence_idx;

        // Range filter
        const double r = node.translation.norm();
        if (r >= g_msgloc.minRangeM && r <= g_msgloc.maxRangeM) {
            const gtsam::Key nid = gtsam::Symbol('s', static_cast<size_t>(i));
            source_graph.addNode(nid, node);
            // ------- Multi-label: store concept, score, and color_rgb -------
            std::map<std::string, std::pair<std::vector<uint8_t>, float>> label_map;

            // Labels and scores: 1:1 match if lengths align; otherwise use overlap range
            std::vector<std::string> concepts;
            if (det.contains("concept") && det["concept"].is_array()) {
                for (const auto& c : det["concept"]) concepts.emplace_back(c.get<std::string>());
            }

            // score accepts both array and scalar formats
            std::vector<float> scores;
            if (det.contains("score")) {
                if (det["score"].is_array()) {
                    for (const auto& s : det["score"]) scores.emplace_back(static_cast<float>(s.get<double>()));
                } else if (det["score"].is_number()) {
                    // If scalar, apply the same score to all labels
                    float s = static_cast<float>(det["score"].get<double>());
                    scores.assign(concepts.size(), s);
                }
            }

            // Handle count mismatch
            const size_t N = std::min(concepts.size(), scores.size());
            if (concepts.size() != scores.size() && !concepts.empty() && !scores.empty()) {
                ROS_WARN("concept/score size mismatch: concepts=%zu scores=%zu (using min=%zu)",
                        concepts.size(), scores.size(), N);
            }

            // Populate map
            for (size_t j = 0; j < N; ++j) {
                label_map.emplace(concepts[j], std::make_pair(std::vector<uint8_t>{}, scores[j]));
            }

            // Store per-node key (order matters)
            if (!label_map.empty()) {
                // 1) Normalize first (read from original map)
                auto norm = normalize_label_map(label_map);

                // 2) Store normalized result
                g_msgloc.SourceLabelCandidates[nid] = std::move(norm);  // move is fine

                // No need to persist the original per-node label map.
            }
        }
    }

    // --- Adjacent edges: nearest k neighbors ---
    // nodes is std::map, so iterate by key order
    std::vector<std::pair<gtsam::Key, QuadricNode>> node_list;
    node_list.reserve(source_graph.nodes.size());
    for (const auto& kv : source_graph.nodes) node_list.push_back(kv);

    for (size_t ai = 0; ai < node_list.size(); ++ai) {
        const auto& [ka, na] = node_list[ai];

        std::vector<std::pair<double, gtsam::Key>> dists;
        dists.reserve(node_list.size()-1);
        for (size_t bi = 0; bi < node_list.size(); ++bi) {
            if (ai == bi) continue;
            const auto& [kb, nb] = node_list[bi];
            const double d = (na.translation - nb.translation).norm();
            dists.emplace_back(d, kb);
        }
        std::sort(dists.begin(), dists.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });

        const int K = std::min<int>(g_msgloc.kNeighbors, static_cast<int>(dists.size()));
        for (int t=0; t<K; ++t) {
            const double d = dists[t].first;
            const gtsam::Key kb = dists[t].second;
            source_graph.addEdge(ka, kb, d);
        }
    }

    return std::make_tuple(source_graph, stamp);
}

void loadjson() {
    std::ifstream ifs(g_msgloc.detection_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + g_msgloc.detection_path);
    }
    all_json = json::parse(ifs);

    if (!all_json.contains("sequences") || !all_json["sequences"].is_array()) {
        throw std::runtime_error("Invalid JSON: 'sequences' not found");
    }

    std::cout << "JSON loaded from " << g_msgloc.detection_path
                << " with " << all_json["sequences"].size()
                << " sequences available." << std::endl;
}

void updateResultValues() {
    for (const auto& node_pair : loadedData.graph.nodes) {
        gtsam::Key key = node_pair.first;
        const QuadricNode& node = node_pair.second;

        // Build ConstrainedDualQuadric from node translation and radii
        gtsam::Vector3 translation(node.translation[0], node.translation[1], node.translation[2]);
        gtsam::Vector3 radii(node.radii[0], node.radii[1], node.radii[2]);

        // orientation is quaternion-form and must be converted properly
        gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(node.orientation[0], node.orientation[1], node.orientation[2], node.orientation[3]);

        // Create ConstrainedDualQuadric
        gtsam_quadrics::ConstrainedDualQuadric quadric(rotation, translation, radii);

        // Insert quadric into result_values
        result_values.insert(key, quadric);

        ROS_INFO("Quadric with key %lu added to result_values.", gtsam::Symbol(key).index());
    }
}

void setupCameraInfo() {
    sensor_msgs::CameraInfo cam_info;

    // Set camera resolution
    cam_info.height = g_msgloc.height;  // Image height
    cam_info.width = g_msgloc.width;   // Image width

    // Distortion model and coefficients
    cam_info.distortion_model = "plumb_bob";  // Distortion model
    cam_info.D = {g_msgloc.d_k1, g_msgloc.d_k2, g_msgloc.d_p1, g_msgloc.d_p2, g_msgloc.d_k3};   // Distortion coefficients

    // Set camera matrix K (3x3 matrix)
    cam_info.K = {g_msgloc.f_x, 0.0, g_msgloc.c_x,   // Row 1: fx, skew, cx
                  0.0, g_msgloc.f_y, g_msgloc.c_y,   // Row 2: 0, fy, cy
                  0.0, 0.0, 1.0};      // Row 3: 0, 0, 1 (fixed)

    // Set rotation matrix R (3x3 matrix)
    cam_info.R = {1.0, 0.0, 0.0,  // Row 1: 1, 0, 0
                  0.0, 1.0, 0.0,  // Row 2: 0, 1, 0
                  0.0, 0.0, 1.0}; // Row 3: 0, 0, 1 (identity matrix)

    // Set projection matrix P (3x4 matrix)
    cam_info.P = {g_msgloc.f_x, 0.0, g_msgloc.c_x, 0.0,   // Row 1: fx, skew, cx, Tx
                  0.0, g_msgloc.f_y, g_msgloc.c_y, 0.0,   // Row 2: 0, fy, cy, Ty
                  0.0, 0.0, 1.0, 0.0};      // Row 3: 0, 0, 1, 0 (fixed)

    // Binning settings
    cam_info.binning_x = 0;
    cam_info.binning_y = 0;

    // ROI settings (Region of Interest)
    cam_info.roi.x_offset = 0;
    cam_info.roi.y_offset = 0;
    cam_info.roi.height = 0;
    cam_info.roi.width = 0;
    cam_info.roi.do_rectify = false;

    // Apply CameraInfo to PinholeCameraModel
    cam_model.fromCameraInfo(cam_info);

    // Also create gtsam calibration as a pointer
    calibration = boost::make_shared<gtsam::Cal3_S2>(g_msgloc.f_x, g_msgloc.f_y, g_msgloc.skew, g_msgloc.c_x, g_msgloc.c_y);

    // Print results
    ROS_INFO("cam_info.K[0]: %f, cam_info.K[4]: %f, cam_info.K[2]: %f, cam_info.K[5]: %f",
             cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]);
    ROS_INFO("fx: %f, fy: %f, cx: %f, cy: %f", cam_model.fx(), cam_model.fy(), cam_model.cx(), cam_model.cy());
}

void publishMapMarkers() {
    if (!map_object_marker_pub_ || !map_edge_marker_pub_) return;

    const ros::Time stamp = ros::Time::now();
    visualization_msgs::MarkerArray object_marker_array;
    visualization_msgs::MarkerArray edge_marker_array;
    object_marker_array.markers.reserve(loadedData.graph.nodes.size() + 1);
    edge_marker_array.markers.reserve(2);
    object_marker_array.markers.push_back(makeDeleteAllMarker(stamp));
    edge_marker_array.markers.push_back(makeDeleteAllMarker(stamp));

    int marker_id = 0;
    for (const auto& kv : loadedData.graph.nodes) {
        const QuadricNode& node = kv.second;

        visualization_msgs::Marker marker;
        marker.header.frame_id = rviz_frame_id_;
        marker.header.stamp = stamp;
        marker.ns = "msgloc_map_nodes";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = node.translation.x();
        marker.pose.position.y = node.translation.y();
        marker.pose.position.z = node.translation.z();
        marker.pose.orientation.x = node.orientation[1];
        marker.pose.orientation.y = node.orientation[2];
        marker.pose.orientation.z = node.orientation[3];
        marker.pose.orientation.w = node.orientation[0];

        marker.scale.x = markerScaleFromRadius(node.radii.x());
        marker.scale.y = markerScaleFromRadius(node.radii.y());
        marker.scale.z = markerScaleFromRadius(node.radii.z());

        if (!applyBaselineColor(kv.first, marker)) {
            marker.color.r = 0.15;
            marker.color.g = 0.85;
            marker.color.b = 0.45;
            marker.color.a = 0.65;
        }
        marker.lifetime = ros::Duration(0.0);
        object_marker_array.markers.push_back(marker);
    }

    visualization_msgs::Marker edge_marker;
    edge_marker.header.frame_id = rviz_frame_id_;
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "msgloc_map_edges";
    edge_marker.id = 0;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;
    edge_marker.action = visualization_msgs::Marker::ADD;
    edge_marker.scale.x = 0.015;
    edge_marker.color.r = 0.05;
    edge_marker.color.g = 0.85;
    edge_marker.color.b = 0.35;
    edge_marker.color.a = 0.75;
    edge_marker.lifetime = ros::Duration(0.0);

    for (const auto& edge : loadedData.graph.edges) {
        const auto it1 = loadedData.graph.nodes.find(edge.first.first);
        const auto it2 = loadedData.graph.nodes.find(edge.first.second);
        if (it1 == loadedData.graph.nodes.end() || it2 == loadedData.graph.nodes.end()) continue;

        edge_marker.points.push_back(makePoint(it1->second.translation.x(),
                                               it1->second.translation.y(),
                                               it1->second.translation.z()));
        edge_marker.points.push_back(makePoint(it2->second.translation.x(),
                                               it2->second.translation.y(),
                                               it2->second.translation.z()));
    }

    if (!edge_marker.points.empty()) edge_marker_array.markers.push_back(edge_marker);

    map_object_marker_pub_.publish(object_marker_array);
    map_edge_marker_pub_.publish(edge_marker_array);
    ROS_INFO("Published map objects to /msgloc_map_objects and edges to /msgloc_map_edges (nodes=%zu, edges=%zu, frame=%s)",
             loadedData.graph.nodes.size(),
             loadedData.graph.edges.size(),
             rviz_frame_id_.c_str());
}

void publishEstimatedPose(const gtsam::Pose3& pose, double stamp_sec, int frame_index) {
    const ros::Time stamp = (std::isfinite(stamp_sec) && stamp_sec > 0.0)
                                ? ros::Time(stamp_sec)
                                : ros::Time::now();

    const gtsam::Point3 t = pose.translation();
    const gtsam::Quaternion q = pose.rotation().toQuaternion();

    nav_msgs::Odometry odom;
    odom.header.frame_id = rviz_frame_id_;
    odom.header.stamp = stamp;
    odom.child_frame_id = pose_child_frame_id_;
    odom.pose.pose.position.x = t.x();
    odom.pose.pose.position.y = t.y();
    odom.pose.pose.position.z = t.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    pose_odom_pub_.publish(odom);

    const auto R = pose.rotation().matrix();
    const double arrow_len = 0.35;
    geometry_msgs::Point start = makePoint(t.x(), t.y(), t.z());
    geometry_msgs::Point end = makePoint(t.x() + arrow_len * R(0, 2),
                                         t.y() + arrow_len * R(1, 2),
                                         t.z() + arrow_len * R(2, 2));

    visualization_msgs::Marker arrow;
    arrow.header.frame_id = rviz_frame_id_;
    arrow.header.stamp = stamp;
    arrow.ns = "msgloc_pose_arrows";
    arrow.id = pose_marker_id_++;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.points.push_back(start);
    arrow.points.push_back(end);
    arrow.scale.x = 0.035;
    arrow.scale.y = 0.10;
    arrow.scale.z = 0.12;
    arrow.color.r = 1.0;
    arrow.color.g = 0.25;
    arrow.color.b = 0.05;
    arrow.color.a = 0.95;
    arrow.lifetime = ros::Duration(0.0);
    pose_arrow_markers_.push_back(arrow);
    pose_path_points_.push_back(start);

    visualization_msgs::Marker path;
    path.header.frame_id = rviz_frame_id_;
    path.header.stamp = stamp;
    path.ns = "msgloc_pose_path";
    path.id = 0;
    path.type = visualization_msgs::Marker::LINE_STRIP;
    path.action = visualization_msgs::Marker::ADD;
    path.scale.x = 0.025;
    path.color.r = 1.0;
    path.color.g = 0.85;
    path.color.b = 0.05;
    path.color.a = 0.95;
    path.lifetime = ros::Duration(0.0);
    path.points = pose_path_points_;

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.reserve(pose_arrow_markers_.size() + 2);
    marker_array.markers.push_back(makeDeleteAllMarker(stamp));
    marker_array.markers.insert(marker_array.markers.end(),
                                pose_arrow_markers_.begin(),
                                pose_arrow_markers_.end());
    if (pose_path_points_.size() >= 2) marker_array.markers.push_back(path);
    pose_marker_pub_.publish(marker_array);

    ROS_INFO("Published estimated pose marker for frame %d to /msgloc_pose_markers (total poses=%zu)",
             frame_index,
             pose_arrow_markers_.size());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// Common utility functions //////////////////////////////////////////

///////////////////////////////// MSG-Loc specific functions /////////////////////////////////
std::pair<double, int> multi_label_likelihood(const gtsam::Key& sourcekey, const gtsam::Key& targetkey) {
    const auto& sourceColorsMap = g_msgloc.SourceLabelCandidates[sourcekey];
    const auto& targetColorsMap = loadedData.TargetLabelLikelihood[targetkey];

    double mll = 0.0;
    int matchedColorsCount = 0;

    for (const auto& sourceLabel : sourceColorsMap) {
        const std::string& sourceLabelName = sourceLabel.first;
        const float sourceScore = sourceLabel.second.second;

        const auto targetLabelIt = targetColorsMap.find(sourceLabelName);
        if (targetLabelIt == targetColorsMap.end()) continue;

        const std::vector<float>& targetData = targetLabelIt->second;
        if (targetData.empty()) continue;

        // Legacy format: targetData[2], new format (map.json): targetData[0]
        const float targetLikelihood = (targetData.size() > 2) ? targetData[2] : targetData[0];
        mll += sourceScore * targetLikelihood;

        if (++matchedColorsCount == g_msgloc.maxLabelMatches) break; // Process up to configured matches
    }

    return {mll, matchedColorsCount};
}

///////////////////////////////////// graph matching ///////////////////////////////////////
//// OURS (M-LLE + CALP)
std::vector<std::pair<gtsam::Key, gtsam::Key>> match_Subgraphs(QuadricGraph& sourceGraph, QuadricGraph& targetGraph) {
    int topk = g_msgloc.matchTopK; // Select top scoring target keys
    std::vector<std::pair<gtsam::Key, gtsam::Key>> result;

    // Final_scoreList is sorted by TotalSim
    std::multimap<double, std::pair<gtsam::Key, gtsam::Key>> Final_scoreList;
    double distanceThreshold = g_msgloc.matchDistanceThreshold;
    for (auto& [sourcekey, sourcenode] : sourceGraph.nodes) {
        for (auto& [targetkey, targetnode] : targetGraph.nodes) {

            // Compute Totalpair.first
            std::pair<double, int> Totalpair = multi_label_likelihood(sourcekey, targetkey);
            double TotalScore = Totalpair.first;
            int validLabelCount = Totalpair.second;  // Assume this is the valid label count returned by multi_label_likelihood

            if (validLabelCount < g_msgloc.minValidLabelCount) {
                continue;
            }
            // Compute average neighbor-node similarity
            double neighborSimSum = 0.0;
            int validNeighborPairs = 0;

            for (const auto& sourceNeighbor : sourcenode.neighbor_node) {
                double maxNeighborSim = 0.0;
                double sourceDistance = sourceGraph.edges[std::make_pair(sourcekey, sourceNeighbor)];
                for (const auto& targetNeighbor : targetnode.neighbor_node) {
                    // Compute source-target neighbor similarity
                    
                    double targetDistance = targetGraph.edges[std::make_pair(targetkey, targetNeighbor)];
                    if (std::abs(sourceDistance - targetDistance) > distanceThreshold) {
                        continue; // Skip if distance is not similar enough
                    }
                    double diff_weight = 1.0 / (1.0 + std::abs(sourceDistance - targetDistance));
                    // Distance similarity condition

                    std::pair<double, int> neighborSim = multi_label_likelihood(sourceNeighbor, targetNeighbor);

                    // Keep the highest similarity
                    if (neighborSim.first * diff_weight > maxNeighborSim) {
                        maxNeighborSim = neighborSim.first * diff_weight ;
                    }
                }

                if (maxNeighborSim >= 0.0) { // If valid similarity exists
                    neighborSimSum += maxNeighborSim;
                    ++validNeighborPairs;
                }
            }

            // If valid neighbors exist, compute mean
            double neighborSimAvg = (validNeighborPairs > 0) ? (neighborSimSum / validNeighborPairs) : 0.0;

            // Compute final similarity
            double TotalSim = (TotalScore + neighborSimAvg) / 2.0;

            if (TotalSim >= 0.0) {
                Final_scoreList.emplace(TotalSim, std::make_pair(sourcekey, targetkey));
            }
        }

        // Extract top-k for each source key
        int count = 0;
        for (auto it = Final_scoreList.rbegin(); it != Final_scoreList.rend() && count < topk; ++it) {

        
            result.emplace_back(it->second);
            ++count;
        }

        // Clear Final_scoreList before the next source key
        Final_scoreList.clear();
    }

    // Print results
    // std::cout << "=== Matching Results (SourceKey -> TargetKey) ===" << std::endl;
    // for (const auto& [sourceKey, targetKey] : result) {
    //     std::cout << "SourceKey: " << gtsam::Symbol(sourceKey).index()
    //             << " -> TargetKey: " << gtsam::Symbol(targetKey).index() << std::endl;
    // }

    return result;
}

////////////////////////////////////  Pose estimation  //////////////////////////////////////
double gaussian_wasserstein_2d(const gtsam_quadrics::AlignedBox2& target, const gtsam_quadrics::AlignedBox2& source) {
    // Compute target center
    Eigen::Vector2f target_min(target.xmin(), target.ymin());
    Eigen::Vector2f target_max(target.xmax(), target.ymax());
    Eigen::Vector2f target_center = (target_min + target_max) / 2.0f;

    // Compute source center
    Eigen::Vector2f source_min(source.xmin(), source.ymin());
    Eigen::Vector2f source_max(source.xmax(), source.ymax());
    Eigen::Vector2f source_center = (source_min + source_max) / 2.0f;
    // Compute target sigma
    double target_width = target.xmax() - target.xmin();
    double target_height = target.ymax() - target.ymin();
    // Compute source sigma
    double source_width = source.xmax() - source.xmin();
    double source_height = source.ymax() - source.ymin();
    // Precompute target sigma diagonal terms
    double target_var_x = target_width * target_width / 4.0;
    double target_var_y = target_height * target_height / 4.0;
    Eigen::Matrix2d sigma1;
    sigma1 << target_var_x, 0,
            0, target_var_y;

    // Precompute source sigma diagonal terms
    double source_var_x = source_width * source_width / 4.0;
    double source_var_y = source_height * source_height / 4.0;
    Eigen::Matrix2d sigma2;
    sigma2 << source_var_x, 0,
            0, source_var_y;

    // Directly compute sqrt(sigma1), sqrt(sigma2) via diagonal terms
    Eigen::Matrix2d sigma11;
    sigma11 << std::sqrt(target_var_x), 0,
            0, std::sqrt(target_var_y);


    // Optimized computation of s121 and sigma121
    Eigen::Matrix2d s121 = sigma11 * sigma2 * sigma11;
    Eigen::Matrix2d sigma121;
    sigma121 << std::sqrt(s121(0, 0)), 0,
                0, std::sqrt(s121(1, 1));

    double d = (target_center.cast<double>() - source_center.cast<double>()).squaredNorm()
            + (sigma1 + sigma2 - 2 * sigma121).trace();
    return d;
}

double normalized_gaussian_wasserstein_2d(const gtsam_quadrics::AlignedBox2& target, const gtsam_quadrics::AlignedBox2& source, const double constant_C)
{
    return exp(-sqrt(gaussian_wasserstein_2d(target, source))/constant_C);
}

// P3P RANSAC 
std::tuple<std::vector<std::pair<gtsam::Key, gtsam::Key>>, gtsam::Pose3, int>
poseEstimation(const QuadricGraph& sourceGraph,
            const QuadricGraph& targetGraph,
            const std::vector<std::pair<gtsam::Key, gtsam::Key>>& result,
            const boost::shared_ptr<gtsam::Cal3_S2>& calibration) {

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << g_msgloc.f_x, g_msgloc.skew, g_msgloc.c_x, 0, g_msgloc.f_y, g_msgloc.c_y, 0, 0, 1);
    cv::Mat distCoeffs   = cv::Mat::zeros(4, 1, CV_64F);

    const int maxIterations = g_msgloc.RANSAC_iters;

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::set<gtsam::Key> Unique_Source;

    // Prepare correspondences
    for (const auto& kp : result) {
        const auto& sNode = sourceGraph.nodes.at(kp.first);
        const auto& qbox  = sNode.bound;
        float xc = float(0.5 * (qbox.xmin() + qbox.xmax()));
        float yc = float(0.5 * (qbox.ymin() + qbox.ymax()));

        const auto& tNode = targetGraph.nodes.at(kp.second);
        const Eigen::Vector3d c = tNode.translation;

        objectPoints.emplace_back(cv::Point3f(float(c[0]), float(c[1]), float(c[2])));
        imagePoints.emplace_back(cv::Point2f(xc, yc));
        Unique_Source.insert(kp.first);
    }

    // Not feasible with fewer than 3 points
    if (Unique_Source.size() < 3) {
        ROS_ERROR("Pose estimation failed: not enough unique image points");
        return {{}, gtsam::Pose3(), 1};
    }

    std::vector<int> bestInliers;
    std::vector<int> finalMatchingIndex;
    cv::Mat bestRvec, bestTvec;
    double bestScore = std::numeric_limits<double>::lowest(); // Maximize W-score
    int iterationCount = 0;

    // Shared set to prevent duplicate combinations across threads
    std::set<std::vector<int>> usedCombinations;

    omp_set_num_threads(g_msgloc.ompNumThreads);
    #pragma omp parallel
    {
        std::vector<int> localBestInliers;
        std::vector<int> localMatchingIndex;
        cv::Mat localBestRvec, localBestTvec;
        double localBestScore = std::numeric_limits<double>::lowest();

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, int(objectPoints.size()) - 1);

        #pragma omp for
        for (int it = 0; it < maxIterations; ++it) {
            // --- 3-point sample (distinct source keys) ---
            std::vector<int> selected;
            selected.reserve(3);
            std::set<gtsam::Key> pickedKeys;

            while ((int)selected.size() < 3) {
                int idx = uni(rng);
                if (pickedKeys.insert(result[idx].first).second) {
                    selected.push_back(idx);
                }
            }
            std::sort(selected.begin(), selected.end());

            // Prevent duplicate combinations (shared across threads)
            bool dup = false;
            #pragma omp critical(usedCombinations)
            {
                if (!usedCombinations.insert(selected).second) {
                    dup = true;
                }
            }
            if (dup) continue;

            // --- Connectivity check (3 points -> 3 pairs) ---
            int totalSrcEdges = 0, matchedEdges = 0;
            for (int i = 0; i < 2; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    auto [s1, t1] = result[selected[i]];
                    auto [s2, t2] = result[selected[j]];
                    bool sConn = sourceGraph.has_edge(s1, s2);
                    bool tConn = targetGraph.has_edge(t1, t2);
                    totalSrcEdges += int(sConn);
                    matchedEdges  += int(sConn && tConn);
                }
            }
            if (totalSrcEdges == 0 || double(matchedEdges) / totalSrcEdges < 1.0) {
                #pragma omp atomic
                ++iterationCount;
                continue;
            }

            // Build sample points
            std::vector<cv::Point3f> obj3(3);
            std::vector<cv::Point2f> img3(3);
            for (int k = 0; k < 3; ++k) {
                obj3[k] = objectPoints[selected[k]];
                img3[k] = imagePoints[selected[k]];
            }

            // --- P3P (multiple solutions) ---
            std::vector<cv::Mat> rvecs, tvecs;
            int nsol = cv::solveP3P(obj3, img3, cameraMatrix, distCoeffs, rvecs, tvecs, cv::SOLVEPNP_P3P);
            if (nsol <= 0) {
                #pragma omp atomic
                ++iterationCount;
                continue;
            }

            // Select the best solution in this iteration (max score)
            double iterBestScore = std::numeric_limits<double>::lowest();
            std::vector<int> iterBestInliers;
            std::vector<int> iterBestMatch;
            int iterBestIdx = -1;

            for (int s = 0; s < nsol; ++s) {
                // Convert to camera pose (world)
                cv::Mat R; cv::Rodrigues(rvecs[s], R);
                R = R.t();
                cv::Mat t = -R * tvecs[s];

                Eigen::Matrix3d Re; cv::cv2eigen(R, Re);
                gtsam::Rot3 gR(Re);
                gtsam::Point3 gT(t.at<double>(0), t.at<double>(1), t.at<double>(2));
                gtsam::Pose3 pose(gR, gT);

                // Evaluate W-score (larger is better)
                std::vector<int> inliers;
                std::vector<int> matchIndex;
                double totalW = 0.0;

                for (const auto& srcKey : Unique_Source) {
                    const auto& srcBox = sourceGraph.nodes.at(srcKey).bound;

                    double bestW = 0.0;
                    int    bestIdx = -1;

                    for (int r = 0; r < (int)result.size(); ++r) {
                        if (result[r].first != srcKey) continue;

                        auto tgtKey = result[r].second;
                        auto quad   = result_values.at<gtsam_quadrics::ConstrainedDualQuadric>(tgtKey);

                        // Front-facing (visibility) check
                        auto qC = quad.pose().translation();
                        auto cP = pose.translation();
                        auto cZ = pose.rotation().matrix().col(2);
                        if (cZ.dot(qC - cP) <= 0) continue;

                        auto proj = gtsam_quadrics::QuadricCamera::project(quad, pose, calibration).bounds();
                        double W  = normalized_gaussian_wasserstein_2d(proj, srcBox, g_msgloc.wassersteinConstant);

                        if (W > bestW) { bestW = W; bestIdx = r; }
                    }

                    totalW += bestW;
                    matchIndex.push_back(bestIdx);
                    if (bestW > g_msgloc.poseInlierThreshold && bestIdx != -1) inliers.push_back(bestIdx);
                }

                totalW /= std::max<size_t>(1, Unique_Source.size());

                if (totalW > iterBestScore) {
                    iterBestScore = totalW;
                    iterBestInliers = std::move(inliers);
                    iterBestMatch   = std::move(matchIndex);
                    iterBestIdx     = s;
                }
            } // for each solution

            // Compare iteration best with local best
            if (iterBestIdx >= 0 && iterBestScore > localBestScore) {
                localBestScore = iterBestScore;
                localBestInliers = std::move(iterBestInliers);
                localMatchingIndex = std::move(iterBestMatch);
                // Store original rvec/tvec
                localBestRvec = rvecs[iterBestIdx].clone();
                localBestTvec = tvecs[iterBestIdx].clone();
            }

            #pragma omp atomic
            ++iterationCount;
        } // omp for

        // Update global best
        #pragma omp critical
        {
            if (localBestScore > bestScore) {
                bestScore = localBestScore;
                bestInliers.swap(localBestInliers);
                finalMatchingIndex.swap(localMatchingIndex);
                bestRvec = localBestRvec.clone();
                bestTvec = localBestTvec.clone();
            }
        }
    } // omp parallel

    std::cout << "Total RANSAC iterations: " << iterationCount << std::endl;

    if ((int)bestInliers.size() < 3) {
        ROS_ERROR("Pose estimation failed: not enough inliers");
        return {{}, gtsam::Pose3(), 2};
    }

    // Build final pose
    cv::Mat R; cv::Rodrigues(bestRvec, R);
    R = R.t();
    cv::Mat t = -R * bestTvec;

    Eigen::Matrix3d Re; cv::cv2eigen(R, Re);
    gtsam::Rot3 gR(Re);
    gtsam::Point3 gT(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    gtsam::Pose3 estPose(gR, gT);

    std::vector<std::pair<gtsam::Key, gtsam::Key>> inlierMatches;
    for (int idx : bestInliers) inlierMatches.push_back(result[idx]);

    return {inlierMatches, estPose, 0};
}

void run_sequence_offline(size_t si, int start_fi = 0, int end_fi = -1) {
    if (!all_json.contains("sequences") || !all_json["sequences"].is_array())
        throw std::runtime_error("run_sequence_offline: invalid all_json.sequences");

    const auto& seqs = all_json["sequences"];
    if (si >= seqs.size())
        throw std::runtime_error("run_sequence_offline: si out of range");

    const auto& seq = seqs[si];
    setSequenceContext(seq);

    if (!seq.contains("frames") || !seq["frames"].is_array()) {
        ROS_WARN("run_sequence_offline: sequence has no frames");
        return;
    }

    const auto& frames_json = seq["frames"];
    const int total_frames = static_cast<int>(frames_json.size());
    if (end_fi < 0 || end_fi >= total_frames) end_fi = total_frames - 1;
    start_fi = std::max(0, start_fi);
    end_fi   = std::max(start_fi, end_fi);

    // --- Output path (pose results only) ---
    const std::string pose = joinPath(g_msgloc.output_dir, "pose_results.txt");

    auto ensure_parent_dir = [](const std::string& path){
        namespace fs = std::filesystem;
        fs::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);
        }
    };
    ensure_parent_dir(pose);

    // --- Open streams once outside the loop ---
    const bool first_open = (save_count == 0);
    const auto output_mode =
        first_open ? (std::ios::out | std::ios::trunc) : (std::ios::out | std::ios::app);
    auto open_output_stream = [output_mode](const std::string& path) {
        return std::ofstream(path, output_mode);
    };

    std::ofstream pose_file_ofs = open_output_stream(pose);

    if (first_open) {
        if (pose_file_ofs.is_open()) {
            pose_file_ofs << "#Timestamp,x,y,z,qx,qy,qz,qw\n";
        }
    }
    
    for (int fi = start_fi; fi <= end_fi; ++fi) {
        const auto& frame = frames_json[fi];

        // ===== 1) Build source graph =====
        const double t0 = ros::Time::now().toSec();
        auto [sourceGraph, stamp] = build_source_graph_from_frame(frame);

        const double image_timestamp = stamp;

        const double t1 = ros::Time::now().toSec();
        const double graph_gen_time = t1 - t0;

        // ===== 2) Matching =====
        const double t2a = ros::Time::now().toSec();
        // MSG-Loc matching
        std::vector<std::pair<gtsam::Key, gtsam::Key>> candidate_key_match =
                        match_Subgraphs(sourceGraph, loadedData.graph);

        const double t2b = ros::Time::now().toSec();
        const double match_time = t2b - t2a;

        // ===== 3) Pose estimation =====
        gtsam::Pose3 current_pose;  // estimated pose
        int sign = 2;

        const double t3a = ros::Time::now().toSec();
        auto pose_result = poseEstimation(sourceGraph, loadedData.graph, candidate_key_match, calibration);
        current_pose = std::get<1>(pose_result);
        sign = std::get<2>(pose_result);
        const double t3b = ros::Time::now().toSec();
        const double pose_refine_time = t3b - t3a;


        // --- success/failure counters only ---
        if (sign == 0) {
            Success_Count++;
        } else if (sign == 1) {
            Insufficient_Count++;
        } else {
            Failure_Count++;
        }
        Total_Count++;

        // ===== 4) per-frame timing summary (EN) =====
        const double total_time = graph_gen_time + match_time + pose_refine_time;

        // --- Aggregate per-stage timings in StageTimer (ALL/SUCC/INSUF/FAIL) ---
        T_graph.add(graph_gen_time,  sign);
        T_match.add(match_time,      sign);
        T_pose.add(pose_refine_time, sign);
        T_total.add(total_time,      sign);

        ROS_INFO(
            "Frame %d | Graph: %.6f s (ALL %.6f | SUCC %.6f) | Match: %.6f s (ALL %.6f | SUCC %.6f) | "
            "Pose: %.6f s (ALL %.6f | SUCC %.6f) | Total: %.6f s (ALL %.6f | SUCC %.6f)%s",
            fi,
            graph_gen_time,   T_graph.mean_all(), T_graph.mean_succ(),
            match_time,       T_match.mean_all(), T_match.mean_succ(),
            pose_refine_time, T_pose.mean_all(),  T_pose.mean_succ(),
            total_time,       T_total.mean_all(), T_total.mean_succ(),
            (sign==0 ? " [OK]" : (sign==1 ? " [SRC<3]" : " [INLIER<3]"))
        );

        // Next line: cumulative counters (EN)
        ROS_INFO(
            "Progress | processed: %d | failures: %d | successes: %d",
            Total_Count, Failure_Count, Success_Count
        );
        // Save pose results: write only successful frames (sign==0)
        if (sign == 0) {
            publishEstimatedPose(current_pose, image_timestamp, fi);
        }

        if (sign == 0 && pose_file_ofs) {
            const gtsam::Point3 t = current_pose.translation();
            const gtsam::Quaternion q = current_pose.rotation().toQuaternion();
            pose_file_ofs << std::fixed << std::setprecision(9)
                            << image_timestamp << " "
                            << t.x() << " " << t.y() << " " << t.z() << " "
                            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            pose_file_ofs.flush();
        }

        save_count++;
        Pose_id++;
        } // for frames

    // Streams are automatically closed at scope exit
    }

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "backend_node");
    ros::NodeHandle pnh("~");

    std::string config_path = g_msgloc.config_path;
    const bool has_config = pnh.getParam("config", config_path);
    if (!has_config && pnh.getParam("comfig", config_path)) {
        ROS_WARN("Using private param '~comfig'. Prefer '~config' if this was a typo.");
    }
    g_msgloc.loadFromYaml(config_path);
    g_msgloc.loadRosPathParams(pnh);

    Msgloc msgloc;

    msgloc.run_sequence_offline(static_cast<size_t>(g_msgloc.sequenceIndex), g_msgloc.startFrame, g_msgloc.endFrame);

    bool rviz_keep_alive = false;
    pnh.param<bool>("rviz_keep_alive", rviz_keep_alive, rviz_keep_alive);
    if (rviz_keep_alive) {
        ROS_INFO("rviz_keep_alive=true: keeping backend_node alive for RViz subscribers.");
        ros::spin();
    }

    return 0;
}
