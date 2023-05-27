FROM registry.gitlab.inria.fr/mgenet/mec-581-repo-2-docker:latest

# Copy home directory for usage in binder
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV HOME /home/${NB_USER}
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

# USER ${NB_USER}
# ENTRYPOINT []
