#include <pybind11/pybind11.h>
#include <dolfinx_hdg/hello.h>

PYBIND11_MODULE(cpp, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("say_hello", &hello::say_hello, "A function which says hello");
}