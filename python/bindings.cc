#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/publish/frame_publisher.h"
#include "openvslam/publish/map_publisher.h"
#include "ndarray_converter.h"
#include "openvslam/type.h"

#include <pybind11/stl.h>
#include "pybind11/pybind11.h"
#include <opencv2/core/core.hpp>
#include <yaml-cpp/yaml.h>

namespace py = pybind11;

PYBIND11_MODULE(openvslam_python, m) {
	NDArrayConverter::init_numpy();

	py::class_<openvslam::config, std::shared_ptr<openvslam::config>>(m, "config")
		.def(py::init<const std::string&>(), py::arg("config_file_path"))
		.def(py::init<const YAML::Node&, const std::string&>(), py::arg("yaml_node"), py::arg("config_file_path") = "");

	py::class_<openvslam::system>(m, "system")
		.def(py::init<const std::shared_ptr<openvslam::config>&, const std::string&>(), py::arg("cfg"), py::arg("vocab_file_path"))
		.def("startup", &openvslam::system::startup, py::arg("need_initialize") = true)
		.def("shutdown", &openvslam::system::shutdown)
		.def("save_frame_trajectory", &openvslam::system::save_frame_trajectory, py::arg("path"), py::arg("format"))
		.def("save_keyframe_trajectory", &openvslam::system::save_keyframe_trajectory, py::arg("path"), py::arg("format"))
		.def("load_map_database", &openvslam::system::load_map_database, py::arg("path"))
		.def("save_map_database", &openvslam::system::save_map_database, py::arg("path"))
		.def("get_map_publisher", &openvslam::system::get_map_publisher)
		.def("get_frame_publisher", &openvslam::system::get_frame_publisher)
		.def("enable_mapping_module", &openvslam::system::enable_mapping_module)
		.def("disable_mapping_module", &openvslam::system::disable_mapping_module)
		.def("mapping_module_is_enabled", &openvslam::system::mapping_module_is_enabled)
		.def("enable_loop_detector", &openvslam::system::enable_loop_detector)
		.def("disable_loop_detector", &openvslam::system::disable_loop_detector)
		.def("loop_detector_is_enabled", &openvslam::system::loop_detector_is_enabled)
		.def("loop_BA_is_running", &openvslam::system::loop_BA_is_running)
		.def("abort_loop_BA", &openvslam::system::abort_loop_BA)
		.def("feed_monocular_frame", &openvslam::system::feed_monocular_frame, py::arg("img"), py::arg("timestamp"), py::arg("mask")= cv::Mat{})
		.def("feed_stereo_frame", &openvslam::system::feed_stereo_frame, py::arg("left_img"), py::arg("right_img"), py::arg("timestamp"), py::arg("mask")= cv::Mat{})
		.def("feed_RGBD_frame", &openvslam::system::feed_RGBD_frame, py::arg("rgb_img"), py::arg("depthmap"), py::arg("timestamp"), py::arg("mask")= cv::Mat{})
		.def("pause_tracker", &openvslam::system::pause_tracker)
		.def("tracker_is_paused", &openvslam::system::tracker_is_paused)
		.def("resume_tracker", &openvslam::system::resume_tracker)
		.def("request_reset", &openvslam::system::request_reset)
		.def("reset_is_requested", &openvslam::system::reset_is_requested)
		.def("request_terminate", &openvslam::system::request_terminate)
		.def("terminate_is_requested", &openvslam::system::terminate_is_requested);
}
