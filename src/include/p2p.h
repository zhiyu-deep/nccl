/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

struct ncclP2Pinfo {
 const void* sendbuff;
  void* recvbuff;
  ssize_t sendbytes;
  ssize_t recvbytes;
};

// todo: 此处先猜测: 一共有nranks peer节点, 但是当前节点最多建立MAXCHANNELS个节点连接(可以和同个节点建立多次连接).
//        link0 link1 link2 ... linkn
//  node0
//  node1
//  ...
//  nodek
struct ncclP2PConnect {
  int nrecv[MAXCHANNELS];
  int nsend[MAXCHANNELS];
  int* recv;  // todo: recv = new int[MAXCHANNELS * nranks];
  int* send;  // todo: send = new int[MAXCHANNELS * nranks];
};

struct ncclP2Plist {
  struct ncclP2Pinfo *peerlist;  // todo: 用来表示和其他卡的连接, peerlist = new ncclP2Pinfo[nranks];
  int count;
  struct ncclP2PConnect connect;
};

#endif
