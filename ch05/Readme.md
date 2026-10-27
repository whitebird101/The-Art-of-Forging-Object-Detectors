# 万物 | 炼器  —— 阵图法则：YAML 蓝图与 `parse_model` 动态引擎

- [YAML 蓝图与 `parse_model` 动态引擎（一）](./YAML蓝图与parse_model动态引擎（一）.md)
  
  介绍YAML蓝图基础参数、模型缩放系数及层配置格式，逐步定义Backbone、Head、跨层连接和相关模块，为后续实现parse_model动态构建模型奠定基础。

- [YAML 蓝图与 `parse_model` 动态引擎（二）](./YAML蓝图与parse_model动态引擎（二）.md)

  详解基于 YAML 动态构建模型的核心机制，通过反射定位模块并解析参数，按模块类型推导通道并处理重复次数，逐层实例化模块、记录结构信息与缓存索引，实现完整模型自动创建。

- [YAML 蓝图与 `parse_model` 动态引擎（三）](./YAML蓝图与parse_model动态引擎（一）.md)

  将 parse_model() 封装进 DetectionModel，依据 from 调度层间连接，通过 save 按需缓存特征，利用虚拟输入验证模型结构与多尺度输出。



