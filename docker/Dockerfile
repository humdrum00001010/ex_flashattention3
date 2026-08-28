# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04@sha256:e3fd439288969c0c78d52768cda4a10ab20f408180d6fa2933f94f7af7810cb0

ARG DEBIAN_FRONTEND=noninteractive
ARG MISE_VERSION=2026.8.14
ARG ERLANG_VERSION=29.0.2
ARG ELIXIR_VERSION=1.20.2-otp-29
ARG NX_COMMIT=37901d749105076e2882fdea89a6e18393eecc0f
ARG FA3_COMMIT=0251105a2fb19d2957484b7f023cd8c115286ced
ARG CUTLASS_COMMIT=7127592069c2fe01b041e174ba4345ef9b279671

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      curl \
      gdb \
      git \
      libnvshmem3-cuda-13=3.3.24-1 \
      locales \
      make \
      openssh-server \
      rsync \
      xz-utils && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen C.UTF-8 && \
    ssh-keygen -A

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:${LD_LIBRARY_PATH} \
    ELIXIR_ERL_OPTIONS="+fnu +sssdio 128" \
    MISE_DATA_DIR=/opt/mise \
    MISE_CACHE_DIR=/opt/mise-cache \
    MISE_STATE_DIR=/opt/mise-state \
    MIX_HOME=/opt/mix \
    HEX_HOME=/opt/hex \
    REBAR_CACHE_DIR=/opt/rebar \
    XLA_TARGET=cuda13 \
    XLA_CACHE_DIR=/opt/fa3-tp/cache/xla \
    NX_OPERATION_ATTRIBUTES_WORKTREE=/opt/fa3-tp/nx-worktree \
    FA3_TP_DYLIB=/opt/fa3-tp/artifacts/libfa3_xla.so \
    CXX=clang++ \
    CC=clang

RUN curl -fsSL https://mise.run | MISE_VERSION="${MISE_VERSION}" sh && \
    install -m 0755 /root/.local/bin/mise /usr/local/bin/mise && \
    mkdir -p \
      "${MISE_DATA_DIR}" "${MISE_CACHE_DIR}" "${MISE_STATE_DIR}" \
      "${MIX_HOME}" "${HEX_HOME}" "${REBAR_CACHE_DIR}" \
      /opt/fa3-tp/cache/xla /opt/fa3-tp/artifacts && \
    mise use --global "erlang@${ERLANG_VERSION}" && \
    ERL_AFLAGS='+JMsingle true' mise use --global "elixir@${ELIXIR_VERSION}"

ENV PATH="/opt/mise/installs/erlang/29.0.2/bin:/opt/mise/installs/elixir/1.20.2-otp-29/bin:${PATH}"

WORKDIR /opt/fa3-tp

RUN git clone --filter=blob:none https://github.com/elixir-nx/nx.git nx-worktree && \
    git -C nx-worktree checkout "${NX_COMMIT}"

COPY nx-overlay/ /opt/fa3-tp/nx-worktree/

RUN git -C nx-worktree diff --check

RUN mkdir -p build && \
    git clone --filter=blob:none https://github.com/Dao-AILab/flash-attention.git build/flash-attention && \
    git -C build/flash-attention checkout "${FA3_COMMIT}" && \
    git -C build/flash-attention submodule update --init --depth 1 csrc/cutlass && \
    test "$(git -C build/flash-attention/csrc/cutlass rev-parse HEAD)" = "${CUTLASS_COMMIT}"

COPY experiment/ /opt/fa3-tp/fa3_tp_experiment/

WORKDIR /opt/fa3-tp/fa3_tp_experiment

RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" ERL_AFLAGS='+JMsingle true' \
      mix local.hex --force && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" ERL_AFLAGS='+JMsingle true' \
      mix deps.get && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" ERL_AFLAGS='+JMsingle true' MIX_ENV=test \
      mix compile && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" ERL_AFLAGS='+JMsingle true' MIX_ENV=test \
      mix test

RUN LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
      LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" \
      make -C native CXX=clang++ -j2 && \
    test -x /opt/fa3-tp/artifacts/libfa3_xla.so && \
    cuobjdump --list-elf /opt/fa3-tp/artifacts/libfa3_xla.so | \
      grep -c '\.sm_90a\.cubin' | grep -qx 8 && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1

COPY experiment/docker/start.sh /start.sh
COPY experiment/docker/fa3-test /usr/local/bin/fa3-test
COPY experiment/docker/fa3-bench /usr/local/bin/fa3-bench

RUN chmod 0755 /start.sh /usr/local/bin/fa3-test /usr/local/bin/fa3-bench && \
    printf '%s\n' \
      'PermitRootLogin prohibit-password' \
      'PubkeyAuthentication yes' \
      'PermitUserEnvironment yes' \
      >> /etc/ssh/sshd_config

WORKDIR /opt/fa3-tp/fa3_tp_experiment
EXPOSE 22
ENTRYPOINT ["/start.sh"]
CMD ["sleep", "infinity"]
