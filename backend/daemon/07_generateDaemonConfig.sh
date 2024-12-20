#!/usr/bin/bash

# Following came from modified settings originally generated by the following
#   snetd init

# References for config files and parameter lists
#   https://github.com/singnet/snet-daemon

USE_EMBEDDED_ETCD_CLUSTER=true
SERVICE_PORT=7002
DAEMON_PORT=8089
#ETCD_DOMAIN=https://127.0.0.1
ETCD_DOMAIN=ai.photrek.io
ETCD_PORT=2379
#Following only applies when using standalone (NOT embedded) ETCD cluster.
ETCD_CLUSTER_DESC="member-1=https://localhost:1180,member-2=https://localhost:1280,member-3=https://localhost:1380"
RUN_WITH_BLOCKCHAIN_ENABLED=false



# Configuration string (used to build daemon config file)
cs='{\n'

#----------------------------------------------------------------------
# Daemon IP:port
#----------------------------------------------------------------------
# Following: port on which daemon listens for incoming client requests
# Note: server-IP must be contained in the config json (e.g., snetd-server.
cs+='  "daemon_end_point": "0.0.0.0:'$DAEMON_PORT'",\n'
#cs+='  "daemon_end_point": "localhost:8088",\n'

# Following: OPTIONAL -- endpt to get service configuration metadata
cs+='  "ipfs_end_point": "http://ipfs.singularitynet.io:80",\n'
# Note: per https://dev.singularitynet.io/docs/ai-developers/troubleshooting/,# we should ideally have our own project ID on infura (how?).


#----------------------------------------------------------------------
# Service activation and IP:port
#----------------------------------------------------------------------
# Following: if false, requests echoed back as responses (for testing)
#cs+='  "passthrough_enabled": false,\n'
cs+='  "passthrough_enabled": true,\n'

# Following is service location (to be proxied if service_type != executable)
cs+='  "passthrough_endpoint": "http://localhost:'$SERVICE_PORT'",\n'

cs+='  "organization_id": "Photrek",\n'
cs+='  "service_id": "cvae",\n'


#----------------------------------------------------------------------
# ETCD configuration
#----------------------------------------------------------------------
cs+='  "payment_channel_storage_type": "etcd",\n'
if false; then
    # Following is per the tutorial conventions
    cs+='  "payment_channel_cert_path": "/home/blake_anderton/snetd_test/ca.pem",\n'
    cs+='  "payment_channel_ca_path": "/home/blake_anderton/snetd_test/ca.pem",\n'
    cs+='  "payment_channel_key_path": "/home/blake_anderton/snetd_test/snetd-client-key.pem",\n'

else
    if $USE_EMBEDDED_ETCD_CLUSTER; then
        # Following is per the GitHub documentation (latest snetd).
        cs+='  "payment_channel_storage_client": {\n'
        cs+='    "connection_timeout": "5s",\n'
        cs+='    "endpoints": [\n'
        #Following used when attempting daemon-internal etcd cluster.
        cs+='      "http://127.0.0.1:12381"\n'
        cs+='    ],\n'
        cs+='    "request_timeout": "3s"\n'
        cs+='  },\n'
       
        cs+='  "payment_channel_storage_server": {\n'
        #cs+='    "client_port": 2379,\n'
        cs+='    "client_port": 12381,\n'
        cs+='    "cluster": "storage-1=http://127.0.0.1:12382",\n'
        cs+='    "data_dir": "storage-data-dir-1.etcd",\n'
        cs+='    "enabled": true,\n'
        cs+='    "host": "127.0.0.1",\n'
        cs+='    "id": "storage-1",\n'
        cs+='    "log_level": "info",\n'
        cs+='    "peer_port": 12382,\n'
        cs+='    "scheme": "http",\n'
        cs+='    "startup_timeout": "1m",\n'
        cs+='    "token": "unique-token"\n'
        cs+='  },\n'

        #cs+='  "payment_channel_cert_path": "",\n'
        #cs+='  "payment_channel_ca_path": "",\n'
        #cs+='  "payment_channel_key_path": "",\n'

        #cs+='  "payment_channel_cert_path": "/var/lib/etcd/cfssl/client.pem",\n'
        #cs+='  "payment_channel_ca_path": "/var/lib/etcd/cfssl/ca.pem",\n'
        #cs+='  "payment_channel_key_path": "/var/lib/etcd/cfssl/client-key.pem",\n'

        #cs+='  "paymen/t_channel_cert_path": "/home/blake_anderton/snetd_test/client.pem",\n'
        #cs+='  "payment_channel_ca_path": "/home/blake_anderton/snetd_test/ca.pem",\n'
        #cs+='  "payment_channel_key_path": "/home/blake_anderton/snetd_test/client-key.pem",\n'
        
    else
        # For following, reference config in ~/etcd/standalone/12_startServer.sh
        # This pertains to utilizing an **existent ETCD cluster**.

        # Following is per the GitHub documentation (latest snetd).
        cs+='  "payment_channel_storage_client": {\n'
        cs+='    "connection_timeout": "5s",\n'
        cs+='    "endpoints": [\n'
        # Following setting exclusive to 3-member etcd cluster.
        cs+='      "'$ETCD_DOMAIN':'$ETCD_PORT'"\n'
        cs+='    ],\n'
        cs+='    "request_timeout": "3s"\n'
        cs+='  },\n'
       
        cs+='  "payment_channel_storage_server": {\n'
        cs+='    "client_port": '$ETCD_PORT',\n'
        cs+='    "cluster": "'$ETCD_CLUSTER_DESC'",\n'
        cs+='    "data_dir": "/var/lib/etcd/server",\n'
        # Following flag indicates whether the **embedded etcd cluster** is enabled.
        cs+='    "enabled": false,\n'
        cs+='    "host": "'$ETCD_DOMAIN'",\n'
        cs+='    "id": "server",\n'
        cs+='    "log_level": "info",\n'
        #cs+='    "peer_port": 2380,\n'
        cs+='    "scheme": "http",\n'
        cs+='    "startup_timeout": "1m",\n'
        cs+='    "token": "unique-token"\n'
        cs+='  },\n'

        # Note: following are for TLS/SSL-secured comm with ETCD.
        # These should be pre-verified (before daemon launch) through curl calls against ETCD cluster.
        # E.g.,  curl -v --cacert /var/lib/etcd/cfssl/ca.pem --cert /var/lib/etcd/cfssl/client.pem --key /var/lib/etcd/cfssl/client-key.pem https://0.0.0.0:2379/health

        cs+='  "payment_channel_cert_path": "/var/lib/etcd/cfssl/client.pem",\n'
        cs+='  "payment_channel_ca_path": "/var/lib/etcd/cfssl/ca.pem",\n'
        cs+='  "payment_channel_key_path": "/var/lib/etcd/cfssl/client-key.pem",\n'
        
    fi
fi

#----------------------------------------------------------------------
# Daemon TLS/SSL-secured comm cert/key
#----------------------------------------------------------------------
# Following attempts (found unsuccessful) to use self-signed snetd certs
#cs+='  "ssl_cert": "/home/blake_anderton/snetd_test/snetd-client.pem",\n'
#cs+='  "ssl_key": "/home/blake_anderton/snetd_test/snetd-client-key.pem",\n'

# Following uses CA-signed fullchain.pem & privkey.pem per following's conventions:
#   https://dev.singularitynet.io/docs/ai-developers/daemon-setup/
cs+='  "ssl_cert": "/etc/letsencrypt/live/ai.photrek.io/fullchain.pem",\n'
cs+='  "ssl_key": "/etc/letsencrypt/live/ai.photrek.io/privkey.pem",\n'

#----------------------------------------------------------------------
# Blockchain activation & specification
#----------------------------------------------------------------------
# Following: if false, used for testing.
cs+='  "blockchain_enabled": '$RUN_WITH_BLOCKCHAIN_ENABLED',\n'
#cs+='  "blockchain_enabled": true,\n'

cs+='  "blockchain_network_selected": "main",\n'


#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
cs+='  "log": {\n'
cs+='    "formatter": {\n'
cs+='      "timestamp_format": "2006-01-02T15:04:05.999999999Z07:00",\n'
cs+='      "type": "text"\n'
cs+='    },\n'
cs+='    "hooks": [],\n'
cs+='    "level": "info",\n'
cs+='    "output": {\n'
cs+='      "current_link": "./snet-daemon.log",\n'
cs+='      "file_pattern": "./snet-daemon.%%Y%%m%%d.log",\n'
cs+='      "max_age_in_sec": 604800,\n'
cs+='      "rotation_count": 0,\n'
cs+='      "rotation_time_in_sec": 86400,\n'
cs+='      "type": "file"\n'
cs+='    },\n'
cs+='    "timezone": "UTC"\n'
cs+='  }\n'
cs+='}' 
#printf `echo $cs` > snetd-config.json
#printf `echo $cs` > snetd-config.json
#echo -e $cs > snetd-config.json
printf "$cs" > snetd-config.json
if [ $? -eq 0 ]; then
    echo "Created: snetd-config.json"
fi

# Following is OPTIONAL
#"ethereum_json_rpc_endpoint": "https://ropsten.infura.io/v3/<YourRegisterdinfuraiID>"


# Following is default values from "snetd init"
# 
#{
#  "alerts_email": "",
#  "allowed_user_flag": false,
#  "auto_ssl_cache_dir": ".certs",
#  "auto_ssl_domain": "",
#  "blockchain_enabled": true,
#  "blockchain_network_selected": "local",
#  "daemon_end_point": "127.0.0.1:8080",
#  "daemon_group_name": "default_group",
#  "daemon_type": "grpc",
#  "enable_dynamic_pricing": false,
#  "hdwallet_index": 0,
#  "hdwallet_mnemonic": "",
#  "ipfs_end_point": "http://localhost:5002/",
#  "ipfs_timeout": 30,
#  "log": {
#    "formatter": {
#      "timestamp_format": "2006-01-02T15:04:05.999999999Z07:00",
#      "type": "text"
#    },
#    "hooks": [],
#    "level": "info",
#    "output": {
#      "current_link": "./snet-daemon.log",
#      "file_pattern": "./snet-daemon.%Y%m%d.log",
#      "max_age_in_sec": 604800,
#      "rotation_count": 0,
#      "rotation_time_in_sec": 86400,
#      "type": "file"
#    },
#    "timezone": "UTC"
#  },
#  "max_message_size_in_mb": 4,
#  "metering_enabled": false,
#  "organization_id": "ExampleOrganizationId",
#  "passthrough_enabled": true,
#  "payment_channel_storage_client": {
#    "connection_timeout": "5s",
#    "endpoints": [
#      "http://127.0.0.1:2379"
#    ],
#    "request_timeout": "3s"
#  },
#  "payment_channel_storage_server": {
#    "client_port": 2379,
#    "cluster": "storage-1=http://127.0.0.1:2380",
#    "data_dir": "storage-data-dir-1.etcd",
#    "enabled": false,
#    "host": "127.0.0.1",
#    "id": "storage-1",
#    "log_level": "info",
#    "peer_port": 2380,
#    "scheme": "http",
#    "startup_timeout": "1m",
#    "token": "unique-token"
#  },
#  "payment_channel_storage_type": "etcd",
#  "private_key": "",
#  "service_heartbeat_type": "http",
#  "service_id": "ExampleServiceId",
#  "ssl_cert": "",
#  "ssl_key": "",
#  "token_expiry_in_minutes": 1440




#true}