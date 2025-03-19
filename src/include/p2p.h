/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

// todo: 连接对象, 和连接对象交互需要用到的信息.
struct ncclP2Pinfo {
 const void* sendbuff;
  void* recvbuff;
  ssize_t sendbytes;  // todo: 默认情况下为-1.
  ssize_t recvbytes;  // todo: 默认情况下为-1.
};

// todo: 当前卡的连接情况(拓扑结构).
struct ncclP2PConnect {
  int nrecv[MAXCHANNELS];
  int nsend[MAXCHANNELS];
	// todo: 每个节点最多建立MAXCHANNELS channels, 此处预分配channel数: MAXCHANNELS * nranks(即假设MAXCHANNELS全都连接某一个peer).
  int* recv;
  int* send;
};

struct ncclP2Plist {
  struct ncclP2Pinfo *peerlist;   // todo: [nranks], 当前卡连接对象;
  int count;
  struct ncclP2PConnect connect;  // todo: 当前卡的连接情况(拓扑结构).
};

// todo:       link0(ncclP2Pinfo) link1(ncclP2Pinfo) link2(ncclP2Pinfo) ... linkN(ncclP2Pinfo)
//       node0
//       node1
//       ...
//       nodeK
// todo: 当前node
//													CHANNEL0 CHANNEL1 CHANNEL2 ... (MAXCHANNELS)
//			 link0(ncclP2Pinfo)
//			 link1(ncclP2Pinfo)
//			 ...

#endif
