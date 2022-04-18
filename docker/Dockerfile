FROM dealii/dealii:master-focal-root
# https://github.com/dealii/docker-files/blob/master/focal/Dockerfile
# https://hub.docker.com/r/dealii/dealii/tags/

# ====================================================================
# Local build instructions
# ------------------------
# From https://stackoverflow.com/a/44590221
# docker build . -t dealii-weak_forms-full_ci -f ./Dockerfile
# docker run -d dealii-weak_forms-full_ci
# docker run --rm -t -i -v `pwd`:/home/dealii/app dealii-weak_forms-full_ci
# ====================================================================

# Enable default user
ARG USER=dealii
USER $USER
ENV HOME /home/$USER
WORKDIR $HOME

# Project settings
ARG REPO=https://github.com/jppelteret/dealii-weak_forms.git
ARG BRANCH=main
ARG WORKING_DIR=${HOME}/weak_forms
ARG BASE_DIR=${WORKING_DIR}/dealii-weak_forms
ARG BUILD_DIR=${BASE_DIR}/build

# Clone, build and install
RUN mkdir -p ${WORKING_DIR} \
    && cd ${WORKING_DIR} \
    && git clone ${REPO} \
	&& cd ${BASE_DIR} \
	&& git checkout ${BRANCH} \
	&& mkdir -p ${BUILD_DIR}
RUN cd ${BUILD_DIR} \
    && cmake \
	-GNinja \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_INSTALL_PREFIX=/usr/local \
	-DDEAL_II_SOURCE_DIR=/usr/src/dealii \
	-DBUILD_BENCHMARKS=OFF \
	-DBUILD_TESTS=ON \
	..

# Install (required root privilages due to destination being opt)
# This needs an extra step if ctest is to be run in the build directory
# See https://github.com/ninja-build/ninja/issues/1302
RUN cd ${BUILD_DIR} \
    && sudo ninja install \
	&& sudo chown $USER:$USER .ninja_*

# Clean up and archive source files
RUN rm -rf ${BUILD_DIR} \
    && sudo cp -r ${BASE_DIR} /usr/src

# ====================================================================
# Debugging
# ----------
# # Run tests
# RUN cd ${BUILD_DIR} \
#     && ctest --output-on-failure

# # Add some useful packages for debugging
# USER root
# RUN apt-get update && apt-get install -y \
#     nano \
# 	gdb \
#     && rm -rf /var/lib/apt/lists/*
# USER $USER
# ====================================================================
