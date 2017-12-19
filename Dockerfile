FROM ubuntu:xenial

MAINTAINER SelvasAI.MLLab <SelvasAI.MLLab@selvas.com>

ENV DEBIAN_FRONTEND noninteractive

ENV PORT 8080

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

ENV LANG ko_KR.UTF-8
ENV LANGUAGE ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

ADD env /env

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    sed -i 's/archive.ubuntu.com/ftp.daumkakao.com/' /etc/apt/sources.list && \
    buildDeps='gcc g++ make build-essential' && \
    excuteDeps='language-pack-ko python3-dev python3-setuptools python3-pip openjdk-8-jre' && \
    apt-get update && \
    apt-get install -y $buildDeps $excuteDeps --no-install-recommends && \
    pip3 install -r /env/requirements.txt && \
    apt-get purge -y --auto-remove $buildDeps && \
    apt-get autoremove -y && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/*

ADD src /src

EXPOSE 8080

ENTRYPOINT ["/tini", "--"]

CMD ["/bin/sh", "/src/run.sh"]