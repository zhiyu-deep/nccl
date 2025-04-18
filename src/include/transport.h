/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "devcomm.h"
#include "graph.h"
#include "nvmlwrap.h"
#include "core.h"
#include "proxy.h"

#define NTRANSPORTS 3
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2

extern struct ncclTransport ncclTransports[];

// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

// todo: 运行过程中, 维护了每张卡的硬件信息.
struct ncclPeerInfo {
  int rank;
  int cudaDev;  			// todo: 当前卡物理id.
  int gdrSupport;     // todo: 是否支持gdr(和ib网卡memory direct access)
  uint64_t hostHash;  // todo: host哈希.
  uint64_t pidHash;   // todo: 当前进程号哈希.
  dev_t shmDev;				// todo: /dev/shm的设备号, can use that information to decide whether we can use SHM for inter-process
											// 		   communication in a container environment
  int64_t busId;      // todo: pcie bus id.
};

#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

struct ncclTransportComm {
  ncclResult_t (*setup)(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int channelId);
  ncclResult_t (*connect)(struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  ncclResult_t (*free)(void*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[4];
  ncclResult_t (*canConnect)(int*, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend);

#endif
