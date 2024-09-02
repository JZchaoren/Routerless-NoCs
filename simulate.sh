#!/bin/bash

topology="Mesh_XY"
algorithm=1
mesh_rows=4
mode="tornado"
injection_rates=(0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
vcs_per_vnet=4
router_latency=1
link_width_bits=128
output="results_$topology""_algo_$algorithm""_$mode""_N=$mesh_rows.txt"

rm -f $output
echo "link_width_bits=$link_width_bits" >> $output

for injectionrate in "${injection_rates[@]}"
do
    echo "Running simulation with injectionrate = $injectionrate"

    ./build/NULL/gem5.opt \
    configs/example/garnet_synth_traffic.py \
    --link-width-bits=$link_width_bits \
    --vcs-per-vnet=$vcs_per_vnet \
    --router-latency=$router_latency \
    --network=garnet --num-cpus=16 --num-dirs=16 \
    --topology=$topology \
    --routing-algorithm=$algorithm \
    --inj-vnet=0 --synthetic=$mode \
    --sim-cycles=5000000 --injectionrate=$injectionrate \
    #--mesh-rows=$mesh_rows

    rm -f network_stats.txt
    echo > network_stats.txt
    grep "packets_injected::total" m5out/stats.txt | sed 's/system.ruby.network.packets_injected::total\s*/packets_injected = /' >> network_stats.txt
    grep "packets_received::total" m5out/stats.txt | sed 's/system.ruby.network.packets_received::total\s*/packets_received = /' >> network_stats.txt
    grep "average_packet_queueing_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_queueing_latency\s*/average_packet_queueing_latency = /' >> network_stats.txt
    grep "average_packet_network_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_network_latency\s*/average_packet_network_latency = /' >> network_stats.txt
    grep "average_packet_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_latency\s*/average_packet_latency = /' >> network_stats.txt
    grep "average_hops" m5out/stats.txt | sed 's/system.ruby.network.average_hops\s*/average_hops = /' >> network_stats.txt

    packets_received=$(grep "packets_received" network_stats.txt | awk '{print $3}')
    num_cpus=$(grep "num_cpus" m5out/statistic.txt | awk -F= '{print $2}'| tail -n 1)
    sim_cycles=$(grep "sim_cycles" m5out/statistic.txt | awk -F= '{print $2}'| tail -n 1)
    Reception_Rate=$(echo "scale=5; $packets_received / ($num_cpus * 100000)" | bc)
    echo packets_received = $packets_received
    echo num_cpus = $num_cpus
    echo sim_cycles = $sim_cycles
    throughput=$(echo "scale=5; $packets_received / 100000" | bc)
    echo "reception_rate = $Reception_Rate" >> network_stats.txt
    echo "throughput = $throughput" >> network_stats.txt

    > m5out/statistic.txt

    avg_latency=$(grep "average_packet_latency" network_stats.txt | awk '{print $3}')
    average_packet_queueing_latency=$(grep "average_packet_queueing_latency" network_stats.txt | awk '{print $3}')
    average_packet_network_latency=$(grep "average_packet_network_latency" network_stats.txt | awk '{print $3}')
    throughput=$(grep "throughput" network_stats.txt | awk '{print $3}')
    average_hops=$(grep "average_hops" network_stats.txt | awk '{print $3}')
    echo "injectionrate = $injectionrate" >> $output
    echo "average_packet_latency = $avg_latency" >> $output
    echo "average_packet_queueing_latency = $average_packet_queueing_latency" >> $output
    echo "average_packet_network_latency = $average_packet_network_latency" >> $output
    echo "average_hops = $average_hops" >> $output
    echo "reception_rate = $Reception_Rate" >> $output
    echo "throughput = $throughput" >> $output
done

echo "-----------------------------------" >> $output
