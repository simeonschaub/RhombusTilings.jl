FROM rocm/dev-ubuntu-24.04:6.4.2-complete

# Install Julia
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12 && \
    ~/.juliaup/bin/juliaup add release && \
    apt-get update && apt-get install -y libdw1
