#!/bin/bash

ip=`ip -json addr show | jq -r '[.[] | select(.operstate == "UP" and ((.addr_info | length) == 2))][-1] | .addr_info[] | select(.family == "inet") | .local'`
ifname=`ip -json addr show | jq -r '[.[] | select(.operstate == "UP" and ((.addr_info | length) == 2))][-1] | .ifname'`

args=`echo "${@:1}"`
echo 'Args are ' $args
ip_loc=$dir_name/$SLURM_JOB_ID
dir_n="$dir_name/"
mkdir -p $dir_n
pwd
module load nccl_2.6.4-1+cuda10.1
export NCCL_SOCKET_IFNAME=$ifname
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
echo "Using "$ifname

find_port() {
    port=$1
    while [[ `netstat -tulpn | grep ":"$port` ]]; do
      port=$(echo "$port + 1" | bc)
    done
    echo $port
}


if [[ $SLURM_PROCID = 0 ]]; then
    echo "$SLURM_PROCID is a master"
    port=`find_port $((10000 + RANDOM % 2000))`
    echo $ip:$port > $ip_loc
    python3 -m dbqf.main --dist-url $ip:$port --world-size $SLURM_NTASKS --rank $SLURM_PROCID --log-dir $dir_n $args
else
    echo "$SLURM_PROCID is a worker"
    while [ ! -f $ip_loc ]; do sleep 1; done
    ip_port=`cat $ip_loc`
    python3 -m dbqf.main --dist-url $ip_port --world-size $SLURM_NTASKS --rank $SLURM_PROCID --log-dir $dir_n $args
fi