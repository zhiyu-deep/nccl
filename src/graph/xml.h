/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef XML_H_
#define XML_H_

// A few constraints to make the implementation easy
#define MAX_STR_LEN 256
#define MAX_ATTR_COUNT 16
#define MAX_SUBS 32
#define MAX_NODES 1024

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

struct ncclXmlNode {
  char name[MAX_STR_LEN];  // todo: 节点名称.
	// todo: 属性相关.
	//	 nAttrs: 当前节点属性个数.
	//	 attrs: 当前节点属性的具体内容, value可能是不同数据类型, int等.
	//		 1. busId(int): 在当前机器的busId, 该busId并不是总线号, 指的其实是定位一个PCIe设备用到的id，即BDF(bus + device + function)，一个bus上有多个设备，一个设备有多个功能，因此通过BDF就可以定位一个设备.
	//		 2. class
	//		 3. link_speed
	//		 4. link_width
  struct {
    char key[MAX_STR_LEN];
    char value[MAX_STR_LEN];
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;

  int type;
  struct ncclXmlNode* parent;          // todo: 当前节点的父节点.
  struct ncclXmlNode* subs[MAX_SUBS];  // todo: 当前节点的所有子节点.
  int nSubs;                           // todo: 表示节点的子节点个数.
};

// todo: xml树.
//		1. nodes: 预分配了节点数, node不需要再实例化, pcie路径上每个节点就是一个node.
//		2. maxIndex: 最后一个节点的index.
struct ncclXml {
  struct ncclXmlNode nodes[MAX_NODES];
  int maxIndex;
};

/* File functions */
#define NCCL_TOPO_XML_VERSION 1
ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml);
ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml);
#define NCCL_GRAPH_XML_VERSION 1
ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml);

/* Auto-detect functions */
ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode);
ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode);

/**************/
/* XML Struct */
/* Functions  */
/**************/
// todo: 在node中, 根据atteName返回attr index, 1. 找到具体index, 2. -1, 表示未找到.
static ncclResult_t xmlGetAttrIndex(struct ncclXmlNode* node, const char* attrName, int* index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a=0; a<nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN-1) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}
// todo: 找到node中, attrname的属性, 并且返回其属性值; 如果不存在该attrname属性, 则属性值为Null.
static ncclResult_t xmlGetAttr(struct ncclXmlNode* node, const char* attrName, const char** value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  *value = index == -1 ? NULL : node->attrs[index].value;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrStr(struct ncclXmlNode* node, const char* attrName, const char** value) {
  NCCLCHECK(xmlGetAttr(node, attrName, value));
  if (*value == NULL) {
    WARN("Attribute %s of node %s not found", attrName, node->name);
    return ncclInternalError;
  }
  return ncclSuccess;
}
static ncclResult_t xmlGetAttrInt(struct ncclXmlNode* node, const char* attrName, int* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrFloat(struct ncclXmlNode* node, const char* attrName, float* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtof(str, NULL);
  return ncclSuccess;
}

static ncclResult_t xmlFindTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}
// todo: 找到节点名为tagName, 属性名为attrName, 属性值为attrValue的node; 否则返回null node.
static ncclResult_t xmlFindTagKv(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node, const char* attrName, const char* attrValue) {
  *node = NULL;
  for (int i=0; i<xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes+i;
    if (strcmp(n->name, tagName) == 0) {
      const char* value;
      NCCLCHECK(xmlGetAttr(n, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *node = n;
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}
// todo: 在xml node, 设置atteName属性的具体值.
static ncclResult_t xmlSetAttr(struct ncclXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  }
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  return ncclSuccess;
}
// todo: 设置int类型属性内容
//  1. 若不存在该属性名, 则创建该属性
//  2. 设置属性名, 属性内容.
static ncclResult_t xmlSetAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%g", value);
  return ncclSuccess;
}

// todo: 从node节点下, 寻找名为subName的子节点, 未找到则为nullptr.
static ncclResult_t xmlGetSub(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    if (strcmp(node->subs[s]->name, subName) == 0) {
      *sub = node->subs[s];
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

// todo: 针对node, 寻找子节点, 满足条件:1. 子节点名为subName, 2. 子节点attrName属性的值为attrValue, 否则返回nullptr.
static ncclResult_t xmlGetSubKv(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub, const char* attrName, const char* attrValue) {
  *sub = NULL;
  for (int s=0; s<node->nSubs; s++) {
    struct ncclXmlNode* subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {
      const char* value;
      NCCLCHECK(xmlGetAttr(subNode, attrName, &value));
      if (value && strcmp(value, attrValue) == 0) {
        *sub = node->subs[s];
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}
static ncclResult_t xmlGetSubKvInt(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub, const char* attrName, const int attrValue) {
  char strValue[10];
  snprintf(strValue, 10, "%d", attrValue);
  NCCLCHECK(xmlGetSubKv(node, subName, sub, attrName, strValue));
  return ncclSuccess;
}

// todo: 往xml中添加node, xml对象中已经预分配node, 所以选取合适的node当作创建的node返回;
//  1. 为node属性内容进行初始化
//  2. 为node设置parent和节点名.
static ncclResult_t xmlAddNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName, struct ncclXmlNode** sub) {
  if (xml->maxIndex == MAX_NODES) {
    WARN("Error : too many XML nodes (max %d)", MAX_NODES);
    return ncclInternalError;
  }
  struct ncclXmlNode* s = xml->nodes+xml->maxIndex++;
  s->nSubs = 0;
  s->nAttrs = 0;
  *sub = s;
  s->parent = parent;
  if (parent) parent->subs[parent->nSubs++] = s;
  strncpy(s->name, subName, MAX_STR_LEN);
  return ncclSuccess;
}

// Dictionary for STR -> INT conversions. No dictionary size information,
// there needs to be a last element with str == NULL.
struct kvDict {
  const char* str;
  int value;
};

static ncclResult_t kvConvertToInt(const char* str, int* value, struct kvDict* dict) {
  struct kvDict* d = dict;
  while (d->str) {
    if (strncmp(str, d->str, strlen(d->str)) == 0) {
      *value = d->value;
      return ncclSuccess;
    }
    d++;
  }
  INFO(NCCL_GRAPH, "KV Convert to int : could not find value of '%s' in dictionary, falling back to %d", str, d->value);
  *value = d->value;
  return ncclSuccess;
}
static ncclResult_t kvConvertToStr(int value, const char** str, struct kvDict* dict) {
  struct kvDict* d = dict;
  while (d->str) {
    if (value == d->value) {
      *str = d->str;
      return ncclSuccess;
    }
    d++;
  }
  WARN("KV Convert to str : could not find value %d in dictionary", value);
  return ncclInternalError;
}

#endif
