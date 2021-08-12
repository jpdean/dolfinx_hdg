#include <pybind11/pybind11.h>
#include "hello.h"

PYBIND11_MODULE(dolfinx_hdg_python, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("say_hello", &hello::say_hello, "A function which says hello");
}
