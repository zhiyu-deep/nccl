/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"

// todo: nccl so中预提供的net库.
//  1. net库分为socket和IB, 两个库基于基础的网络接口提供了各自的实现;
//  2. 用户层面, 在网络接口层面串联具体的业务逻辑.
extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

// todo: nccl中实际使用的net库.
extern ncclNet_t* ncclNet;

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];  // todo: 其实就是解析成socket.h中的socketAddress.

// todo: 外部include本头文件, 真实对外提供的net接口, 接口内部调用的是ncclNet_t中的接口.
// Translation to external API
static const char* ncclNetName() { return ncclNet->name; }
static ncclResult_t ncclNetDevices(int* ndev) { NCCLCHECK(ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetGetProperties(int dev, ncclNetProperties_t* props) { NCCLCHECK(ncclNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t ncclNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm) { NCCLCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(void* listenComm, void** recvComm) { NCCLCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetRegMr(void* comm, void* data, int size, int type, void** mhandle) { NCCLCHECK(ncclNet->regMr(comm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetDeregMr(void* comm, void* mhandle) { NCCLCHECK(ncclNet->deregMr(comm, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->isend(sendComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->irecv(recvComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetFlush(void* recvComm, void* data, int size, void* mhandle) { NCCLCHECK(ncclNet->flush(recvComm, data, size, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* size) { NCCLCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(void* sendComm) { NCCLCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(void* recvComm) { NCCLCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(void* listenComm) { NCCLCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

/**
 * rdma在通信前需要注册一段内存，使得网卡知道虚拟地址和物理地址的映射.
 * 但是如果每次通信都需要将data从显存拷贝到内存再通信的话效率就比较低.
 * IB提供了peer memory的接口，使得ib网卡可以访问其他PCIe空间,
 * nv基于peer memory实现了自己的驱动，使得rdma可以直接注册显存
 * 这样通信就可以避免host和device的内存拷贝，IB可以直接dma显存，即gdr。
 */
// Test whether the current GPU support GPU Direct RDMA.
#define GPU_BUF_SIZE (2*1024*1024)
static ncclResult_t ncclGpuGdrSupport(int* gdrSupport) {
  int netDevs;
  NCCLCHECK(ncclNetDevices(&netDevs));
  *gdrSupport = 0;
  for (int dev=0; dev<netDevs; dev++) {
    // Find a net device which is GDR-capable
    ncclNetProperties_t props;
    NCCLCHECK(ncclNet->getProperties(dev, &props));

		// todo: 1. IB支持cuda ptr, 表示可以注册显存, 即gdr support.
    if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;

		// todo: 2. 创建rdma连接, 验证是否gdr support.
    // Allocate memory on the GPU and try to register it on the NIC.
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    void* gpuPtr = NULL;
    void* mHandle = NULL;
    NCCLCHECK(ncclNetListen(dev, &handle, &lComm));
    NCCLCHECK(ncclNetConnect(dev, &handle, &sComm));
    NCCLCHECK(ncclNetAccept(lComm, &rComm));
    CUDACHECK(cudaMalloc(&gpuPtr, GPU_BUF_SIZE));
    ncclDebugNoWarn = NCCL_NET;
		// todo: 验证send和read句柄下注册显存正确性.
    if (ncclNetRegMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {
      NCCLCHECK(ncclNetDeregMr(sComm, mHandle));
      NCCLCHECK(ncclNetRegMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));
      NCCLCHECK(ncclNetDeregMr(rComm, mHandle));
      *gdrSupport = 1;
    }
    ncclDebugNoWarn = 0;
    CUDACHECK(cudaFree(gpuPtr));
    NCCLCHECK(ncclNetCloseRecv(rComm));
    NCCLCHECK(ncclNetCloseSend(sComm));
    NCCLCHECK(ncclNetCloseListen(lComm));
    break;
  }
  return ncclSuccess;
}

#endif
