FROM bdcp/base:0.1

ENV FLASK_APP=openeo_driver.views
ENV FLASK_DEBUG=1 

USER root
COPY ./docker-entrypoint.sh /docker-entrypoint.sh

COPY ./ ${HOME_DIR}/openeo-python-driver

RUN cd ${HOME_DIR}/openeo-python-driver \
    &&  pip install -e .[dev] --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-openeo/simple

USER bdc

WORKDIR ${HOME_DIR}/openeo-python-driver

ENTRYPOINT [ "/docker-entrypoint.sh" ]