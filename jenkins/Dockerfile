# Jenkins LTS + Python 3.12 (via uv) so the pipeline can run the ML scripts
# on the built-in node. uv pins the interpreter independent of the Debian
# release the Jenkins image is based on.
FROM jenkins/jenkins:lts-jdk17

USER root
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.12 \
 && chown -R jenkins:jenkins /opt/uv-python
USER jenkins
