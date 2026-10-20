# README

See examples/c_example.cpp for a simple usage of the API and DLL.

The provided executable is a compiled version of the above linked example. For the executable to run it needs to be in the same folder as the provided DLL.

**Good to know**

Bounds-checking is performed for:
* The requested boat speed
* The inputted True Wind Speed (TWS)

These have upper bounds as follows in this version:
* Requested boat speed upper bound: 13 m/s (~25.3 kn)
* Inputted True Wind Speed (TWS) upper bound: 24 m/s

**Known limitations**

* Uses a FP propeller (meaning that rps will never go to zero due to "windmilling") and it is also allowed to go above the nominal rate.
* Wind and waves are assumed to be aligned. Cannot handle swell.

**Publications**
If the provided performance model is used to generate results that will be published, please mention the origin of the model (i.e. SSPA Maritime Center at RISE).
