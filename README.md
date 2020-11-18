# Accelerate GMRES (and other linear solvers) using Machine Learning

Here we demonstrate the usage of online machine learning (using pytorch) to accelerate a GMRES-based CFD solver. The philsophy of this project is to design a machine learning pipeline that will accelerate the time-to-solution provided by existing solver codes with minimal user intervention, while minimizing the size of the training data. By using wrappers/function decorators we have made this approach portable to a broad range of iterative solver (not just GMRES). And by using an online-learning approach, we are simulating only as much data as is needed.

This is very much a work in progress -- we are actively adding features/refactoring experimental code. If you have any questions, please feel free to reach out to Johannes Blaschke: https://www.nersc.gov/about/nersc-staff/data-science-engagement-group/johannes-blaschke/

## Getting Started

The fastest way to get started, would be to try out the [demonstration on speeding up the 2D Poisson problem](Demo.ipynb).


## C++ Version

An (even more) experimental C++ version is available here: https://github.com/ML4FnP/GMRES-Learning-CPP
