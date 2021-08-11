# DOLFINx_HDG

## Installation

### From source

#### C++ core

To build and install the C++ core, in the ``cpp/`` directory, run:
```
mkdir build
cd build
cmake ..
cmake --build .
```

#### Python interface

To install the Python interface, first install the C++ core, and then
in the ``python/`` directory run:
```
pip install .
```
(you may need to use ``pip3``, depending on your system).
