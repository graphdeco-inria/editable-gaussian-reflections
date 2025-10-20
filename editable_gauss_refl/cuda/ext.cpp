#include <string>
#include <torch/extension.h>

struct Config {
    int version;
    std::string name;
    bool debug;

    Config(int version = 1, const std::string &name = "default", bool debug = false)
        : version(version), name(name), debug(debug) {}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Config>(m, "Config")
        .def(
            pybind11::init<int, const std::string &, bool>(),
            pybind11::arg("version") = 1,
            pybind11::arg("name") = "default",
            pybind11::arg("debug") = false)
        .def_readwrite("version", &Config::version)
        .def_readwrite("name", &Config::name)
        .def_readwrite("debug", &Config::debug)
        .def("__repr__", [](const Config &c) {
            return "<Config version=" + std::to_string(c.version) + " name='" + c.name +
                   "' debug=" + (c.debug ? "True" : "False") + ">";
        });
}