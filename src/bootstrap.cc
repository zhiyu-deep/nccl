/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include "socket.h"
#include <unistd.h>
#include <sys/types.h>

// todo: socket句柄.
struct bootstrapNetComm {
  int fd;
};

/* Init functions */
static char bootstrapNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];  // todo: interface名字信息, MAX_IF_NAME_SIZE: 机器上interface最大名字长度, MAX_IFS: 最大interface个数.
static union socketAddress bootstrapNetIfAddrs[MAX_IFS];    // todo: interface address信息(封装了底层的sockaddr类).
static int bootstrapNetIfs = -1;                            // todo: 最后得到的interface个数.
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

// todo: 保存本机所有interface信息, 初始化: 1. bootstrapNetIfNames, 2. bootstrapNetIfAddrs, 3. bootstrapNetIfs.
ncclResult_t bootstrapNetInit() {
  if (bootstrapNetIfs == -1) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetIfs == -1) {
      bootstrapNetIfs = findInterfaces(bootstrapNetIfNames, bootstrapNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (bootstrapNetIfs <= 0) {
        WARN("Bootstrap : no socket interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<bootstrapNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, bootstrapNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&bootstrapNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO(NCCL_INIT, "Bootstrap : Using%s", line);
      }
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

static ncclResult_t bootstrapNetNewComm(struct bootstrapNetComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

static ncclResult_t bootstrapNetGetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= bootstrapNetIfs) return ncclInternalError;
  memcpy(addr, bootstrapNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// todo: 服务端, 创建listen socket句柄.
//    对于netHandle(socketAddress):
//      1. dev: 获取第dev个socket address, 将netHandle更新为该地址.
//      2. findsubnetif: 获取netHandle的子网络.
//      3. dontCareIf: 直接使用传进来的netHandle.
//    listenComm, 返回在监听地址上的监听句柄.
static ncclResult_t bootstrapNetListen(int dev, ncclNetHandle_t* netHandle, void** listenComm) {
  union socketAddress* connectAddr = (union socketAddress*) netHandle;
  static_assert(sizeof(union socketAddress) < NCCL_NET_HANDLE_MAXSIZE, "union socketAddress size is too large");

  // if dev >= 0, listen based on dev
  if (dev >= 0) {
    NCCLCHECK(bootstrapNetGetSocketAddr(dev, connectAddr));
  } else if (dev == findSubnetIf) {
    // handle stores a remote address
    // need to find a local addr that is in the same network as the remote addr
    union socketAddress localAddr;
    char ifName[MAX_IF_NAME_SIZE];
    if (findInterfaceMatchSubnet(ifName, &localAddr, connectAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      WARN("NET/Socket : No usable listening interface found");
      return ncclSystemError;
    }
    // pass the local address back
    memcpy(connectAddr, &localAddr, sizeof(localAddr));
  } // Otherwise, handle stores a local address

  struct bootstrapNetComm* comm;
  NCCLCHECK(bootstrapNetNewComm(&comm));
  NCCLCHECK(createListenSocket(&comm->fd, connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

// todo: 客户端, 创建到netHandle连接的socket句柄.
//    netHandle, connect的目标地址;
//    listenComm, 返回连接到远程目标的句柄.
static ncclResult_t bootstrapNetConnect(int dev, ncclNetHandle_t* netHandle, void** sendComm) {
  union socketAddress* connectAddr = (union socketAddress*) netHandle;
  struct bootstrapNetComm* comm;
  NCCLCHECK(bootstrapNetNewComm(&comm));
  NCCLCHECK(connectAddress(&comm->fd, connectAddr));
  *sendComm = comm;
  return ncclSuccess;
}

// todo: 作为服务器, 接受远端的连接, 并且返回accept句柄.
static ncclResult_t bootstrapNetAccept(void* listenComm, void** recvComm) {
  struct bootstrapNetComm* lComm = (struct bootstrapNetComm*)listenComm;

  struct bootstrapNetComm* rComm;
  NCCLCHECK(bootstrapNetNewComm(&rComm));
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  *recvComm = rComm;
  return ncclSuccess;
}

// todo: 通过socket句柄发送数据, 发送端通过句柄发送, 接收端通过句柄接收.
//    1. sync function.
//    2. 先发送数据大小信息, 再发送具体数据内容.
// Additional sync functions
static ncclResult_t bootstrapNetSend(void* sendComm, void* data, int size) {
  struct bootstrapNetComm* comm = (struct bootstrapNetComm*)sendComm;
  NCCLCHECK(socketSend(comm->fd, &size, sizeof(int)));
  NCCLCHECK(socketSend(comm->fd, data, size));
  return ncclSuccess;
}
static ncclResult_t bootstrapNetRecv(void* recvComm, void* data, int size) {
  struct bootstrapNetComm* comm = (struct bootstrapNetComm*)recvComm;
  int recvSize;
  NCCLCHECK(socketReceive(comm->fd, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d\n", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(socketReceive(comm->fd, data, std::min(recvSize, size)));
  return ncclSuccess;
}

static ncclResult_t bootstrapNetClose(void* opaqueComm) {
  struct bootstrapNetComm* comm = (struct bootstrapNetComm*)opaqueComm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}
static ncclResult_t bootstrapNetCloseSend(void* sendComm) { NCCLCHECK(bootstrapNetClose(sendComm)); return ncclSuccess; }
static ncclResult_t bootstrapNetCloseRecv(void* recvComm) { NCCLCHECK(bootstrapNetClose(recvComm)); return ncclSuccess; }
static ncclResult_t bootstrapNetCloseListen(void* listenComm) { NCCLCHECK(bootstrapNetClose(listenComm)); return ncclSuccess; }

// todo: 基于str(hostname + server)信息, 创建socketAddress(ncclNetHandle_t).
ncclResult_t bootstrapNetCreateHandle(ncclNetHandle_t* netHandle, const char* str) {
  union socketAddress* connectAddr = (union socketAddress*) netHandle;
  NCCLCHECK(GetSocketAddrFromString(connectAddr, str));
  return ncclSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  ncclNetHandle_t extHandleListenRoot;
  ncclNetHandle_t extHandleListen;
};

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static void *bootstrapRoot(void* listenComm) {
  struct extInfo info;
  ncclNetHandle_t *rankHandles = NULL;
  ncclNetHandle_t *rankHandlesRoot = NULL; // for initial rank <-> root information exchange
  ncclNetHandle_t zero = { 0 }; // for sanity checking
  void* tmpComm;
  ncclResult_t res;
  setFilesLimit();

  TRACE(NCCL_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
    NCCLCHECKGOTO(bootstrapNetAccept(listenComm, &tmpComm), res, out);
    NCCLCHECKGOTO(bootstrapNetRecv(tmpComm, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(bootstrapNetCloseRecv(tmpComm), res, out);

    if (c == 0) {
      nranks = info.nranks;
      NCCLCHECKGOTO(ncclCalloc(&rankHandles, nranks), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankHandlesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(&zero, &rankHandlesRoot[info.rank], sizeof(ncclNetHandle_t)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankHandlesRoot+info.rank, info.extHandleListenRoot, sizeof(ncclNetHandle_t));
    memcpy(rankHandles+info.rank, info.extHandleListen, sizeof(ncclNetHandle_t));

    ++c;
    TRACE(NCCL_INIT, "Received connect from rank %d total %d/%d",  info.rank, c, nranks);
  } while (c < nranks);
  TRACE(NCCL_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nranks; ++r) {
    int next = (r+1) % nranks;
    void *tmpSendComm;
    NCCLCHECKGOTO(bootstrapNetConnect(0, rankHandlesRoot+r, &tmpSendComm), res, out);
    NCCLCHECKGOTO(bootstrapNetSend(tmpSendComm, rankHandles+next, sizeof(ncclNetHandle_t)), res, out);
    NCCLCHECKGOTO(bootstrapNetCloseSend(tmpSendComm), res, out);
  }
  TRACE(NCCL_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  bootstrapNetCloseListen(listenComm);
  if (rankHandles) free(rankHandles);
  if (rankHandlesRoot) free(rankHandlesRoot);

  TRACE(NCCL_INIT, "DONE");
  return NULL;
}

// todo: 先获取socketAddrss(ncclNetHandle_t), 然后在socketAddress的基础上创建socket listen;
//       本接口由root调用, root会在listen的基础上, 完成对其他进程的组网工作.
//  idFromEnv:
//    1. true: 表示从env中获取hostname + server信息来创建socketAddrss(ncclNetHandle_t), 并且默认id中已经是创建好的socketAddrss(ncclNetHandle_t), 直接使用.
//    2. false: 则默认使用本地0号address.
//  ncclUniqueId: 更新为最终使用的socketAddress.
ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  ncclNetHandle_t* netHandle = (ncclNetHandle_t*) id;
  void* listenComm;
  NCCLCHECK(bootstrapNetListen(idFromEnv ? dontCareIf : 0, netHandle, &listenComm));

  // todo: root在listen基础上启动线程, 完成对其他进程的组网工作.
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, listenComm);
  return ncclSuccess;
}

// todo: 本接口由root调用, 创建socketAddress(ncclNetHandle_t).
ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  static_assert(sizeof(ncclNetHandle_t) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  memset(id, 0, sizeof(ncclUniqueId));
  ncclNetHandle_t* netHandle = (ncclNetHandle_t*) id;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    // todo: 基于env传入的hostname + server信息, 创建socketAddress(ncclNetHandle_t).
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (bootstrapNetCreateHandle(netHandle, env) != 0) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    // todo: 默认使用本机的0号ip, 创建socketAddress(ncclNetHandle_t);
    //       额外的, 本接口由root调用, root在socketAddress(ncclNetHandle_t)基础上启动监听线程, 完成对其他进程的组网工作.
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  // todo: 补充, env == true情况下, root还未启动子线程, 会在后面流程中再启动.

  return ncclSuccess;
}

struct unexConn {
  int peer;
  void* comm;
  struct unexConn* next;
};

// todo: 构建成功的bootstrap网络状态.
struct extState {
  void* extBstrapListenComm;    // todo: 当前节点的监听socket
  void* extBstrapRingRecvComm;  // todo: 当前节点和prev节点的socket连接
  void* extBstrapRingSendComm;  // todo: 当前节点连接next的socket连接
  ncclNetHandle_t* peerBstrapHandles;  // todo: 所有rank的ip port(对应extBstrapListenComm)
  struct unexConn* unexpectedConnections;
  int rank;
  int nranks;
  int dev;                      // todo: 表示在第几个ip地址监听.
};

ncclResult_t bootstrapInit(ncclUniqueId * id, int rank, int nranks, void** commState) {
  ncclNetHandle_t* netHandle = (ncclNetHandle_t*) id;
  bool idFromEnv = getenv("NCCL_COMM_ID") != NULL;

  struct extState* state;
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  struct extInfo info = { 0 };
  info.rank = rank;
  info.nranks = nranks;
  // Pass the remote address to listen via info
  if (idFromEnv) {
    memcpy(&info.extHandleListen, netHandle, sizeof(ncclNetHandle_t));
    memcpy(&info.extHandleListenRoot, netHandle, sizeof(ncclNetHandle_t));
  }

  // listen will return the local address via info (specify interface type 'findSubnetIf')
  state->dev = idFromEnv ? findSubnetIf : 0;
  void* extBstrapListenCommRoot;
  NCCLCHECK(bootstrapNetListen(state->dev, &info.extHandleListen, &state->extBstrapListenComm));
  NCCLCHECK(bootstrapNetListen(state->dev, &info.extHandleListenRoot, &extBstrapListenCommRoot));

  // stagger connection times to avoid an overload of the root at very high rank counts
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(NCCL_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void) nanosleep(&tv, NULL);
  }

  void *tmpSendComm, *tmpRecvComm;
  // send info on my listening socket to root
  NCCLCHECK(bootstrapNetConnect(state->dev, netHandle, &tmpSendComm));
  NCCLCHECK(bootstrapNetSend(tmpSendComm, &info, sizeof(info)));
  NCCLCHECK(bootstrapNetCloseSend(tmpSendComm));

  // get info on my "next" rank in the bootstrap ring from root
  ncclNetHandle_t extHandleNext;
  NCCLCHECK(bootstrapNetAccept(extBstrapListenCommRoot, &tmpRecvComm));
  NCCLCHECK(bootstrapNetRecv(tmpRecvComm, &extHandleNext, sizeof(extHandleNext)));
  NCCLCHECK(bootstrapNetCloseRecv(tmpRecvComm));
  NCCLCHECK(bootstrapNetCloseListen(extBstrapListenCommRoot));

  NCCLCHECK(bootstrapNetConnect(state->dev, &extHandleNext, &state->extBstrapRingSendComm));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(bootstrapNetAccept(state->extBstrapListenComm, &state->extBstrapRingRecvComm));

  // AllGather all listen handlers
  NCCLCHECK(ncclCalloc(&state->peerBstrapHandles, nranks));
  memcpy(state->peerBstrapHandles+rank, info.extHandleListen, sizeof(ncclNetHandle_t));
  NCCLCHECK(bootstrapAllGather(state, state->peerBstrapHandles, sizeof(ncclNetHandle_t)));

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i=0; i<nranks-1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    NCCLCHECK(bootstrapNetSend(state->extBstrapRingSendComm, data+sslice*size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(state->extBstrapRingRecvComm, data+rslice*size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapSend(void* commState, int peer, void* data, int size) {
  struct extState* state = (struct extState*)commState;
  void* tmpSendComm;
  NCCLCHECK(bootstrapNetConnect(state->dev, state->peerBstrapHandles+peer, &tmpSendComm));
  NCCLCHECK(bootstrapNetSend(tmpSendComm, &state->rank, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(tmpSendComm, data, size));
  NCCLCHECK(bootstrapNetCloseSend(tmpSendComm));
  return ncclSuccess;
}

ncclResult_t unexpectedEnqueue(struct extState* state, int peer, void* comm) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->comm = comm;

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}

void* unexpectedDequeue(struct extState* state, int peer) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  while (elem) {
    if (elem->peer == peer) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      void* comm = elem->comm;
      free(elem);
      return comm;
    }
    prev = elem;
    elem = elem->next;
  }
  return NULL;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, void* data, int size) {
  struct extState* state = (struct extState*)commState;

  void* tmpRecvComm;

  // Search unexpected connections first
  if ((tmpRecvComm = unexpectedDequeue(state, peer)) != NULL) {
    NCCLCHECK(bootstrapNetRecv(tmpRecvComm, ((char*)data), size));
    NCCLCHECK(bootstrapNetCloseRecv(tmpRecvComm));
    return ncclSuccess;
  }

  // Then look for new connections
  while (1) {
    NCCLCHECK(bootstrapNetAccept(state->extBstrapListenComm, &tmpRecvComm));
    int newPeer;
    NCCLCHECK(bootstrapNetRecv(tmpRecvComm, &newPeer, sizeof(int)));
    if (newPeer == peer) {
      NCCLCHECK(bootstrapNetRecv(tmpRecvComm, ((char*)data), size));
      NCCLCHECK(bootstrapNetCloseRecv(tmpRecvComm));
      return ncclSuccess;
    }
    // Unexpected connection. Save for later.
    NCCLCHECK(unexpectedEnqueue(state, newPeer, tmpRecvComm));
  }
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (state->unexpectedConnections != NULL) {
    WARN("Unexpected connections are not empty.\n");
    return ncclInternalError;
  }
  NCCLCHECK(bootstrapNetCloseListen(state->extBstrapListenComm));
  NCCLCHECK(bootstrapNetCloseSend(state->extBstrapRingSendComm));
  NCCLCHECK(bootstrapNetCloseRecv(state->extBstrapRingRecvComm));

  free(state->peerBstrapHandles);
  free(state);

  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  struct extState* state = (struct extState*)commState;
  bootstrapNetCloseListen(state->extBstrapListenComm);
  bootstrapNetCloseSend(state->extBstrapRingSendComm);
  bootstrapNetCloseRecv(state->extBstrapRingRecvComm);
  free(state->peerBstrapHandles);
  free(state);
  return ncclSuccess;
}
