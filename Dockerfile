FROM tensorflow/tensorflow:2.9.1-gpu

ARG CI_BUILD_GID
ARG CI_BUILD_GROUP
ARG CI_BUILD_UID
ARG CI_BUILD_USER
ARG CI_BUILD_HOME=/home/${CI_BUILD_USER}

ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV no_proxy ${no_proxy}

ENV HTTP_PROXY ${HTTP_PROXY}
ENV HTTPS_PROXY ${HTTPS_PROXY}
ENV NO_PROXY ${NO_PROXY}

RUN apt-get update
RUN apt-get install -y sudo

############################# Set same user in container #############################
RUN getent group "${CI_BUILD_GID}" || addgroup --force-badname --gid ${CI_BUILD_GID} ${CI_BUILD_GROUP}
RUN getent passwd "${CI_BUILD_UID}" || adduser --force-badname --gid ${CI_BUILD_GID} --uid ${CI_BUILD_UID} \
      --disabled-password --home ${CI_BUILD_HOME} --quiet ${CI_BUILD_USER}
RUN usermod -a -G sudo ${CI_BUILD_USER}
RUN echo "${CI_BUILD_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-nopasswd-sudo

USER ${CI_BUILD_UID}:${CI_BUILD_GID}

RUN whoami

WORKDIR ${CI_BUILD_HOME}
######################################################################################

ENV PATH ${CI_BUILD_HOME}/bin:$PATH

RUN sudo -E apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        rsync \
        software-properties-common \
        wget \
        git \    
        curl \
        vim \
        less

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8