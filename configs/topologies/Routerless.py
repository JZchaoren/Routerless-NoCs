# Copyright (c) 2010 Advanced Micro Devices, Inc.
#               2016 Georgia Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from m5.params import *
from m5.objects import *
from topologies.ConstructRouterless import *

from common import FileSystemConfig

from topologies.BaseTopology import SimpleTopology

# Creates a generic Mesh assuming an equal number of cache
# and directory controllers.
# XY routing is enforced (using link weights)
# to guarantee deadlock freedom.


class Routerless(SimpleTopology):
    description = "Routerless"

    def __init__(self, controllers):
        self.nodes = controllers

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        N = int(options.num_cpus ** (0.5))  # N*N network
        rings = compose_layers(N)  # rings: [[(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)], [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0)], [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3), (0, 2), (0, 1), (0, 0)], [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2), (1, 1), (1, 0)], [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1), (0, 0)], [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 2), (2, 1), (2, 0)], [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1), (0, 1), (0, 0)], [(0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2), (0, 1)], [(0, 2), (1, 2), (2, 2), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3), (0, 2)]]
        edge_count = count_edges_in_rings(rings)
        # print(rings)
        construct_routing_table(rings, N)

        # 对于重复的边，我们需要更多的router，来作为中间定向。对于每一条重复的边，我们都需要增加k个router
        num_additional_router = 0
        for edge, count in edge_count.items():
            if count != 1:
                num_additional_router += count

        nodes = self.nodes

        # 常规的router数加上多余的router
        num_routers = options.num_cpus + num_additional_router

        link_latency = options.link_latency  # used by simple and garnet
        router_latency = options.router_latency  # only used by garnet

        # Create the routers in the mesh
        routers = [
            Router(router_id=i, latency=router_latency)
            for i in range(num_routers)
        ]
        network.routers = routers

        # link counter to set unique link ids
        link_count = 0

        cntrls_per_router, remainder = divmod(len(nodes), options.num_cpus)
        # Add all but the remainder nodes to the list of nodes to be uniformly
        # distributed across the network.
        network_nodes = []
        remainder_nodes = []
        for node_index in range(len(nodes)):
            if node_index < (len(nodes) - remainder):
                network_nodes.append(nodes[node_index])
            else:
                remainder_nodes.append(nodes[node_index])

        # Connect each node to the appropriate router
        ext_links = []
        for (i, n) in enumerate(network_nodes):
            cntrl_level, router_id = divmod(i, options.num_cpus)
            assert cntrl_level < cntrls_per_router
            ext_links.append(
                ExtLink(
                    link_id=link_count,
                    ext_node=n,
                    int_node=routers[router_id],
                    latency=link_latency,
                )
            )
            link_count += 1

        # Connect the remainding nodes to router 0.  These should only be
        # DMA nodes.
        for (i, node) in enumerate(remainder_nodes):
            assert i < remainder
            ext_links.append(
                ExtLink(
                    link_id=link_count,
                    ext_node=node,
                    int_node=routers[0],
                    latency=link_latency,
                )
            )
            link_count += 1

        network.ext_links = ext_links

        # Create the ring links.
        int_links = []
        bridge = options.num_cpus  # maintain this

        for ring_id, ring in enumerate(rings):
            flag = False
            if ring_id == 3:
                flag = True
            if len(ring) == 0:  # no links, no ports, continue
                continue
            for i in range(len(ring) - 2):
                src = ring[i]
                dst = ring[i + 1]
                edge = (src, dst)
                if edge in edge_count:
                    if edge_count[edge] > 1:
                        int_links.append(
                            IntLink(
                                link_id=link_count,
                                src_node=routers[src[1] * N + src[0]],
                                dst_node=routers[bridge],
                                src_outport=f"ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{i}",
                                dst_inport=f"bridge_ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{i}_globalx{dst[0]}_globaly{dst[1]}_local{i+1}",
                                latency=link_latency,
                                weight=1,
                            )
                        )
                        link_count += 1
                        int_links.append(
                            IntLink(
                                link_id=link_count,
                                src_node=routers[bridge],
                                dst_node=routers[dst[1] * N + dst[0]],
                                src_outport=f"bridge_ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{i}_globalx{dst[0]}_globaly{dst[1]}_local{i+1}",
                                dst_inport=f"ring{ring_id}_globalx{dst[0]}_globaly{dst[1]}_local{i+1}",
                                latency=link_latency,
                                weight=1,
                            )
                        )
                        link_count += 1
                        bridge += 1
                        continue

                int_links.append(
                    IntLink(
                        link_id=link_count,
                        src_node=routers[src[1] * N + src[0]],
                        dst_node=routers[dst[1] * N + dst[0]],
                        src_outport=f"ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{i}",
                        dst_inport=f"ring{ring_id}_globalx{dst[0]}_globaly{dst[1]}_local{i+1}",
                        latency=link_latency,
                        weight=1,
                    )
                )
                link_count += 1

            src = ring[-2]
            dst = ring[0]
            edge = (src, dst)
            if edge in edge_count:
                if edge_count[edge] > 1:
                    int_links.append(
                        IntLink(
                            link_id=link_count,
                            src_node=routers[src[1] * N + src[0]],
                            dst_node=routers[bridge],
                            src_outport=f"ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{len(ring)-2}",
                            dst_inport=f"bridge_ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{len(ring)-2}_globalx{dst[0]}_globaly{dst[1]}_local{0}",
                            latency=link_latency,
                            weight=1,
                        )
                    )
                    link_count += 1
                    int_links.append(
                        IntLink(
                            link_id=link_count,
                            src_node=routers[bridge],
                            dst_node=routers[dst[1] * N + dst[0]],
                            src_outport=f"bridge_ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{len(ring)-2}_globalx{dst[0]}_globaly{dst[1]}_local{0}",
                            dst_inport=f"ring{ring_id}_globalx{dst[0]}_globaly{dst[1]}_local{0}",
                            latency=link_latency,
                            weight=1,
                        )
                    )
                    link_count += 1
                    bridge += 1
                    continue
            int_links.append(
                IntLink(
                    link_id=link_count,
                    src_node=routers[src[1] * N + src[0]],
                    dst_node=routers[dst[1] * N + dst[0]],
                    src_outport=f"ring{ring_id}_globalx{src[0]}_globaly{src[1]}_local{len(ring)-2}",
                    dst_inport=f"ring{ring_id}_globalx{dst[0]}_globaly{dst[1]}_local0",
                    latency=link_latency,
                    weight=1,
                )
            )
            link_count += 1

        network.int_links = int_links


    # Register nodes with filesystem
    def registerTopology(self, options):
        for i in range(options.num_cpus):
            FileSystemConfig.register_node(
                [i], MemorySize(options.mem_size) // options.num_cpus, i
            )
